#[allow(unused_imports)]
use super::*;

use std::cell::RefCell;
use std::rc::Rc;

use euv::wasm_bindgen::{JsCast, JsValue};
use euv::*;

use crate::hook::{CameraOrbit, start_water_loop, use_water_state};

type DomEvent = euv::web_sys::Event;

/// Top-level page. Mounts a full-viewport canvas for the WebGPU water
/// surface and wires the gesture handlers (single-finger orbit, two-finger
/// pinch) to a shared `CameraOrbit` state.
pub(crate) fn app() -> VirtualNode {
    let state: UseWater = use_water_state();
    let orbit: CameraOrbit = CameraOrbit::new();
    let loop_started: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));

    // Kick off the WebGPU init + RAF loop immediately. `spawn_local` returns
    // a future that the wasm executor drives; we don't block the render.
    let state_for_init: UseWater = state.clone();
    let orbit_for_init: CameraOrbit = orbit.clone();
    let loop_started_for_init: Rc<RefCell<bool>> = loop_started.clone();
    euv::wasm_bindgen_futures::spawn_local(async move {
        if !*loop_started_for_init.borrow() {
            *loop_started_for_init.borrow_mut() = true;
            start_water_loop(state_for_init, orbit_for_init);
        }
    });

    // Snapshot reactive values into plain Rust values before entering the
    // html! body. The html! body is a FnMut closure so any reactive
    // signal would be moved into it the first time it's read, leaving
    // later reads dangling.
    //
    // To make sure signal changes re-render the page we expose the
    // underlying signals (not snapshots) via direct `.get()` calls
    // inside html! expression slots — the html! macro auto-unwraps
    // single-segment identifier paths to `.get()`, which registers the
    // signal as a subscriber of this dynamic node.
    let ready_signal: euv::Signal<bool> = state.get_ready();
    let fps_signal: euv::Signal<f32> = state.get_fps();
    let error_message_signal: euv::Signal<String> = state.get_error_message();

    html! {
        div {
            class: c_water_root()
            id: "water-root"
            canvas {
                class: c_water_canvas()
                id: "water-canvas"
                onpointerdown: make_pointer_down_handler(orbit.clone())
                onpointermove: make_pointer_move_handler(orbit.clone())
                onpointerup: make_pointer_up_handler()
                onpointercancel: make_pointer_up_handler()
                onpointerleave: make_pointer_up_handler()
                onwheel: make_wheel_handler(orbit.clone())
            }
            div {
                class: c_water_loading_overlay()
                id: "water-loading"
                if { !ready_signal.get() } {
                    "Initializing WebGPU…"
                }
            }
            div {
                class: c_water_hint()
                "drag to orbit · pinch to zoom"
            }
            div {
                class: c_water_stats()
                span {
                    class: c_water_stats_label()
                    "FPS: "
                }
                span {
                    class: c_water_stats_value()
                    { format!("{:.0}", fps_signal.get()) }
                }
            }
            if !error_message_signal.get().is_empty() {
                div {
                    class: c_water_error_box()
                    { error_message_signal.get() }
                }
            }
        }
    }
}

// ─── Gesture state ───────────────────────────────────────────────────────────

thread_local! {
    static GESTURE: RefCell<GestureState> = const { RefCell::new(GestureState::Idle) };
}

fn make_pointer_down_handler(orbit: CameraOrbit) -> Option<Rc<dyn Fn(DomEvent)>> {
    Some(Rc::new(move |event: DomEvent| {
        let pointers: Vec<(i32, f32, f32)> = collect_pointers(&event);
        if pointers.len() == 1 {
            let (_, x, y) = pointers[0];
            GESTURE.with(|g| {
                *g.borrow_mut() = GestureState::Rotating {
                    last_x: x,
                    last_y: y,
                }
            });
        } else if pointers.len() >= 2 {
            let d: f32 = distance_2d(&pointers[0], &pointers[1]);
            GESTURE.with(|g| {
                *g.borrow_mut() = GestureState::Pinching {
                    initial_distance: d,
                    initial_orbit_distance: orbit.distance.get(),
                }
            });
        }
        event.prevent_default();
    }))
}

fn make_pointer_move_handler(orbit: CameraOrbit) -> Option<Rc<dyn Fn(DomEvent)>> {
    Some(Rc::new(move |event: DomEvent| {
        let pointers: Vec<(i32, f32, f32)> = collect_pointers(&event);
        let state: GestureState = GESTURE.with(|g| *g.borrow());
        match state {
            GestureState::Rotating { last_x, last_y } => {
                if let Some((_, x, y)) = pointers.first() {
                    let dx: f32 = x - last_x;
                    let dy: f32 = y - last_y;
                    orbit.add_yaw(-dx * ROTATE_SENSITIVITY);
                    orbit.add_pitch(-dy * ROTATE_SENSITIVITY);
                    let nx: f32 = *x;
                    let ny: f32 = *y;
                    GESTURE.with(|g| {
                        *g.borrow_mut() = GestureState::Rotating {
                            last_x: nx,
                            last_y: ny,
                        };
                    });
                }
            }
            GestureState::Pinching {
                initial_distance,
                initial_orbit_distance,
            } => {
                if pointers.len() >= 2 {
                    let d: f32 = distance_2d(&pointers[0], &pointers[1]);
                    if initial_distance > 1.0 {
                        let scale: f32 = initial_distance / d.max(1.0);
                        // In scale-space: scale > 1 = fingers closer = zoom out
                        // (multiplicative distance grows).
                        let target_distance: f32 = initial_orbit_distance * scale;
                        let clamped: f32 = target_distance.clamp(MIN_DISTANCE_M, MAX_DISTANCE_M);
                        orbit.distance.set(clamped);
                    }
                }
            }
            GestureState::Idle => {}
        }
        event.prevent_default();
    }))
}

fn make_pointer_up_handler() -> Option<Rc<dyn Fn(DomEvent)>> {
    Some(Rc::new(move |_event: DomEvent| {
        GESTURE.with(|g| *g.borrow_mut() = GestureState::Idle);
    }))
}

fn make_wheel_handler(orbit: CameraOrbit) -> Option<Rc<dyn Fn(DomEvent)>> {
    Some(Rc::new(move |event: DomEvent| {
        // Mouse-wheel fallback for desktop users without a trackpad.
        let delta_y: f32 = js_sys::Reflect::get(&event, &JsValue::from_str("deltaY"))
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        if delta_y.abs() > 0.0 {
            let factor: f32 = (delta_y * PINCH_SENSITIVITY * 0.02).exp();
            orbit.multiply_distance(factor);
        }
        event.prevent_default();
    }))
}

fn collect_pointers(event: &DomEvent) -> Vec<(i32, f32, f32)> {
    let mut out: Vec<(i32, f32, f32)> = Vec::new();
    if let Some(target) = event_target(event)
        && let Ok(offsets) = js_sys::Reflect::get(&target, &JsValue::from_str("touches"))
        && let Ok(touch_list) = offsets.dyn_into::<js_sys::Object>()
    {
        let length: u32 = js_sys::Reflect::get(&touch_list, &JsValue::from_str("length"))
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as u32;
        for i in 0..length {
            let touch: Result<JsValue, JsValue> =
                js_sys::Reflect::get(&touch_list, &JsValue::from_f64(i as f64));
            if let Ok(touch_js) = touch {
                let id: i32 = js_sys::Reflect::get(&touch_js, &JsValue::from_str("identifier"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as i32;
                let x: f32 = js_sys::Reflect::get(&touch_js, &JsValue::from_str("clientX"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                let y: f32 = js_sys::Reflect::get(&touch_js, &JsValue::from_str("clientY"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                out.push((id, x, y));
            }
        }
        return out;
    }
    // PointerEvent fallback (mouse + pointer).
    let id: i32 = js_sys::Reflect::get(event, &JsValue::from_str("pointerId"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as i32;
    let x: f32 = js_sys::Reflect::get(event, &JsValue::from_str("clientX"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let y: f32 = js_sys::Reflect::get(event, &JsValue::from_str("clientY"))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    out.push((id, x, y));
    out
}

fn event_target(event: &DomEvent) -> Option<JsValue> {
    js_sys::Reflect::get(event, &JsValue::from_str("target")).ok()
}

fn distance_2d(a: &(i32, f32, f32), b: &(i32, f32, f32)) -> f32 {
    let dx: f32 = a.1 - b.1;
    let dy: f32 = a.2 - b.2;
    (dx * dx + dy * dy).sqrt()
}

// ─── Styling ─────────────────────────────────────────────────────────────────

class! {
    pub c_water_root {
        position: "fixed";
        inset: "0";
        overflow: "hidden";
        background: "#000";
        touch-action: "none";
    }

    pub c_water_canvas {
        display: "block";
        width: "100%";
        height: "100%";
        touch-action: "none";
    }

    pub c_water_loading_overlay {
        position: "absolute";
        inset: "0";
        display: "flex";
        align-items: "center";
        justify-content: "center";
        color: "rgba(255, 255, 255, 0.65)";
        font-size: "14px";
        letter-spacing: "0.04em";
        background: "rgba(0, 0, 0, 0.75)";
        pointer-events: "none";
        z-index: "5";
    }

    pub c_water_hint {
        position: "absolute";
        left: "50%";
        bottom: "20px";
        transform: "translateX(-50%)";
        color: "rgba(255, 255, 255, 0.55)";
        font-size: "11px";
        letter-spacing: "0.08em";
        text-transform: "uppercase";
        pointer-events: "none";
        z-index: "4";
    }

    pub c_water_stats {
        position: "absolute";
        top: "16px";
        right: "16px";
        display: "flex";
        gap: "6px";
        font-family: "ui-monospace, monospace";
        font-size: "12px";
        color: "rgba(255, 255, 255, 0.75)";
        background: "rgba(0, 0, 0, 0.4)";
        padding: "6px 10px";
        border: "1px solid rgba(255, 255, 255, 0.18)";
        pointer-events: "none";
        z-index: "4";
    }

    pub c_water_stats_label {
        color: "rgba(255, 255, 255, 0.55)";
    }

    pub c_water_stats_value {
        color: "rgba(255, 255, 255, 0.95)";
    }

    pub c_water_error_box {
        position: "absolute";
        top: "50%";
        left: "50%";
        transform: "translate(-50%, -50%)";
        max-width: "min(560px, 90vw)";
        padding: "16px 20px";
        background: "rgba(40, 0, 0, 0.92)";
        color: "#fff";
        font-size: "13px";
        border: "1px solid rgba(255, 96, 96, 0.6)";
        z-index: "20";
    }
}
