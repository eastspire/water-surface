//! All hook functions: state factory, gesture handlers, RAF loop entrypoint.
//!
//! The simulation runs on the CPU (JavaScript-side) so we don't need any
//! of euv-engine's `pub(crate)` GPU primitives. Each frame we integrate
//! the 2D wave equation in a Rust-side `Vec<f32>`, upload the result to a
//! `r32float` texture via `queue.writeTexture`, and then call
//! `render_frame_with_bind_group` to draw the surface.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use euv::wasm_bindgen::prelude::*;
use euv::wasm_bindgen::{JsCast, JsValue};
use euv::wasm_bindgen_futures::spawn_local;
use euv::web_sys::*;
use euv::*;

use euv_engine::*;
use js_sys::{Function, Math, Object, Reflect};

use super::r#const::*;
use super::r#struct::{CameraOrbit, UseWater};
use crate::shader::surface::SURFACE_SHADER;

/// Creates the page-level reactive state.
pub(crate) fn use_water_state() -> UseWater {
    UseWater::default()
}

/// Wraps an arbitrary `Reflect::set` call so it can be chained.
macro_rules! js_set {
    ($obj:expr, $key:expr, $val:expr) => {
        let _ = Reflect::set(&$obj, &$key, &$val);
    };
}

/// Builds a JS dictionary object from a sequence of (key, value) pairs.
fn js_dict<const N: usize>(entries: [(&str, JsValue); N]) -> JsValue {
    let obj = Object::new();
    for (k, v) in entries {
        let _ = Reflect::set(&obj, &JsValue::from_str(k), &v);
    }
    obj.into()
}

/// Creates one of the three ping-pong r32float height textures used by the
/// wave equation. Usage = TEXTURE_BINDING | COPY_DST so the CPU can write
/// ripples into it from JS each frame and the GPU can sample it.

/// Reads the canvas element's CSS layout size.
fn read_canvas_size(selector: &str) -> Option<(f32, f32)> {
    let window_value = window()?;
    let document_value = window_value.document()?;
    let element = document_value.query_selector(selector).ok().flatten()?;
    let canvas: web_sys::HtmlCanvasElement = element.unchecked_into();
    let rect = canvas.get_bounding_client_rect();
    Some((rect.width() as f32, rect.height() as f32))
}

fn normalize3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let len = (x * x + y * y + z * z).sqrt();
    if len < 1e-6 {
        (0.0, 0.0, -1.0)
    } else {
        (x / len, y / len, z / len)
    }
}

fn look_at(eye: (f32, f32, f32), target: (f32, f32, f32), up: (f32, f32, f32)) -> [f32; 16] {
    let f = normalize3(target.0 - eye.0, target.1 - eye.1, target.2 - eye.2);
    let s = normalize3(
        f.1 * up.2 - f.2 * up.1,
        f.2 * up.0 - f.0 * up.2,
        f.0 * up.1 - f.1 * up.0,
    );
    let u = (s.1 * f.2 - s.2 * f.1, s.2 * f.0 - s.0 * f.2, s.0 * f.1 - s.1 * f.0);
    let mut m = [0.0_f32; 16];
    m[0] = s.0;
    m[1] = u.0;
    m[2] = -f.0;
    m[3] = 0.0;
    m[4] = s.1;
    m[5] = u.1;
    m[6] = -f.1;
    m[7] = 0.0;
    m[8] = s.2;
    m[9] = u.2;
    m[10] = -f.2;
    m[11] = 0.0;
    m[12] = -(s.0 * eye.0 + s.1 * eye.1 + s.2 * eye.2);
    m[13] = -(u.0 * eye.0 + u.1 * eye.1 + u.2 * eye.2);
    m[14] = f.0 * eye.0 + f.1 * eye.1 + f.2 * eye.2;
    m[15] = 1.0;
    m
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    let mut m = [0.0_f32; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) * nf;
    m[11] = -1.0;
    m[14] = 2.0 * far * near * nf;
    m
}

fn multiply_mat4(a: &[f32; 16], b: &[f32; 16]) -> Vec<f32> {
    let mut out = vec![0.0_f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0_f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

/// Builds the view-projection matrix.
fn build_view_proj(camera_pos: (f32, f32, f32), w: f32, h: f32) -> Vec<f32> {
    let view = look_at(camera_pos, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0));
    let aspect = w / h.max(1.0);
    let proj = perspective(std::f32::consts::FRAC_PI_4, aspect, 0.5, 1000.0);
    multiply_mat4(&proj, &view)
}

/// Returns the camera's forward unit vector (camera → origin).
fn camera_forward(orbit: &CameraOrbit) -> (f32, f32, f32) {
    let yaw = orbit.yaw.get();
    let pitch = orbit.pitch.get();
    let cy = pitch.cos();
    normalize3(-yaw.sin() * cy, -pitch.sin(), -yaw.cos() * cy)
}

/// Per-frame uniform buffer for the surface render pipeline.
fn build_surface_uniforms(
    view_proj: &[f32],
    camera_pos: (f32, f32, f32),
    forward: (f32, f32, f32),
    time: f32,
    mesh_origin: (f32, f32),
    cell_size: f32,
) -> Vec<f32> {
    let mut u = vec![0.0_f32; 52];
    for i in 0..16 {
        u[i] = view_proj[i];
    }
    u[16] = camera_pos.0;
    u[17] = camera_pos.1;
    u[18] = camera_pos.2;
    u[19] = 1.0;
    u[20] = SUN_DIR_X;
    u[21] = SUN_DIR_Y;
    u[22] = SUN_DIR_Z;
    u[23] = 1.0;
    u[24] = forward.0;
    u[25] = forward.1;
    u[26] = forward.2;
    u[27] = 1.0;
    u[28] = time;
    u[29] = mesh_origin.0;
    u[30] = mesh_origin.1;
    u[31] = cell_size;
    u[32] = GRID_RESOLUTION as f32;
    u[33] = HEIGHT_AMPLITUDE_M;
    u[34] = FRESNEL_POWER;
    u[35] = FRESNEL_REFLECTIVITY;
    u[36] = WATER_COLOR.0;
    u[37] = WATER_COLOR.1;
    u[38] = WATER_COLOR.2;
    u[39] = 0.0;
    u[40] = SUN_COLOR.0;
    u[41] = SUN_COLOR.1;
    u[42] = SUN_COLOR.2;
    u[43] = 0.0;
    u[44] = SKY_TOP_COLOR.0;
    u[45] = SKY_TOP_COLOR.1;
    u[46] = SKY_TOP_COLOR.2;
    u[47] = 0.0;
    u[48] = SKY_HORIZON_COLOR.0;
    u[49] = SKY_HORIZON_COLOR.1;
    u[50] = SKY_HORIZON_COLOR.2;
    u[51] = 0.0;
    u
}

/// Vertex buffer for the GRID_RESOLUTION × GRID_RESOLUTION grid. Each vertex
/// is a 2D position in mesh-local space ranging over [0, MESH_TILE_SIZE_M]^2.
fn build_grid_vertex_buffer() -> Vec<u8> {
    let n = GRID_RESOLUTION as usize;
    let mesh_size = MESH_TILE_SIZE_M;
    let mut data = Vec::with_capacity(n * n * 2 * 4);
    for row in 0..n {
        for col in 0..n {
            let x = (col as f32 / (n - 1) as f32) * mesh_size;
            let z = (row as f32 / (n - 1) as f32) * mesh_size;
            data.extend_from_slice(&x.to_le_bytes());
            data.extend_from_slice(&z.to_le_bytes());
        }
    }
    data
}

/// Index buffer for the grid — 2 triangles per quad.
fn build_grid_index_buffer() -> Vec<u8> {
    let n = GRID_RESOLUTION as usize;
    let mut data = Vec::with_capacity((n - 1) * (n - 1) * 6 * 4);
    for row in 0..(n - 1) {
        for col in 0..(n - 1) {
            let tl = (row * n + col) as u32;
            let tr = tl + 1;
            let bl = tl + n as u32;
            let br = bl + 1;
            data.extend_from_slice(&tl.to_le_bytes());
            data.extend_from_slice(&bl.to_le_bytes());
            data.extend_from_slice(&tr.to_le_bytes());
            data.extend_from_slice(&tr.to_le_bytes());
            data.extend_from_slice(&bl.to_le_bytes());
            data.extend_from_slice(&br.to_le_bytes());
        }
    }
    data
}

// ─── Wave simulation (CPU side) ───────────────────────────────────────────────

/// A CPU-side wave-equation simulation. Two height buffers (prev, curr) per
/// cell; each step integrates `h_new = 2h - prev + c²·∇²h` then applies
/// damping.
struct WaveSimulation {
    prev: Vec<f32>,
    curr: Vec<f32>,
}

impl WaveSimulation {
    fn new() -> Self {
        let n = (GRID_RESOLUTION * GRID_RESOLUTION) as usize;
        Self {
            prev: vec![0.0; n],
            curr: vec![0.0; n],
        }
    }

    fn inject_gaussian(&mut self, cx: i32, cy: i32, amplitude: f32, radius: i32) {
        let n = GRID_RESOLUTION as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 || x >= n || y >= n {
                    continue;
                }
                let r2 = (dx * dx + dy * dy) as f32;
                let sigma2 = (radius as f32 * 0.5).powi(2);
                let bump = amplitude * (-r2 / (2.0 * sigma2)).exp();
                self.curr[(y * n + x) as usize] += bump;
            }
        }
    }

    fn step(&mut self, c_squared: f32, damping: f32) {
        let n = GRID_RESOLUTION as usize;
        let mut next = vec![0.0_f32; n * n];
        for y in 1..(n - 1) {
            for x in 1..(n - 1) {
                let idx = y * n + x;
                let center = self.curr[idx];
                let hm = self.curr[idx - 1];
                let hp = self.curr[idx + 1];
                let hd = self.curr[idx - n];
                let hu = self.curr[idx + n];
                let prev = self.prev[idx];
                let laplacian = hm + hp + hd + hu - 4.0 * center;
                let accel = c_squared * laplacian;
                next[idx] = (2.0 * center - prev + accel) * (1.0 - damping);
            }
        }
        // Boundaries decay toward zero (soft reflective).
        for i in 0..n {
            next[i] = 0.0;
            next[(n - 1) * n + i] = 0.0;
            next[i * n] = 0.0;
            next[i * n + (n - 1)] = 0.0;
        }
        std::mem::swap(&mut self.prev, &mut self.curr);
        self.curr.copy_from_slice(&next);
    }
}

/// Live GPU + CPU resources, owned by the RAF closure.
struct WaterRenderer {
    renderer: WebGpuRenderer,
    render_pipeline: JsValue,
    height_buffer: JsValue,
    surface_uniform: JsValue,
    surface_bind_group: JsValue,
    vertex_buffer: JsValue,
    index_buffer: JsValue,
    /// Number of indices in the index buffer.
    vertex_count: u32,
    mesh_origin: Rc<RefCell<(f32, f32)>>,
    sim: Rc<RefCell<WaveSimulation>>,
    orbit: CameraOrbit,
    cancelled: Rc<Cell<bool>>,
    raf_id: Rc<Cell<Option<i32>>>,
    state: UseWater,
}

impl WaterRenderer {
    fn cell_size() -> f32 {
        MESH_TILE_SIZE_M / GRID_RESOLUTION as f32
    }

    fn snap_mesh_to_camera(&self) {
        let cell = Self::cell_size();
        let cam = self.orbit.camera_position();
        let cx = (cam.0 / cell).round() * cell - MESH_TILE_SIZE_M * 0.5;
        let cz = (cam.2 / cell).round() * cell - MESH_TILE_SIZE_M * 0.5;
        *self.mesh_origin.borrow_mut() = (cx, cz);
    }
}

/// Main entrypoint. Spawns the WebGPU init + RAF loop. Called from the view
/// once on first mount.
pub(crate) fn start_water_loop(state: UseWater, orbit: CameraOrbit) {
    spawn_local(async move {
        match init_water_renderer(state.clone(), orbit.clone()).await {
            Ok(renderer) => {
                let renderer = Rc::new(RefCell::new(renderer));
                run_water_loop(renderer).await;
            }
            Err(err) => {
                let msg = format!("WebGPU init failed: {err:?}");
                web_sys::console::error_1(&JsValue::from_str(&msg));
                state.get_error_message().set(msg);
                state.get_ready().set(true);
            }
        }
    });
}

async fn init_water_renderer(
    state: UseWater,
    orbit: CameraOrbit,
) -> Result<WaterRenderer, JsValue> {
    let config = RenderConfig::webgpu(WATER_CANVAS_SELECTOR, 1280.0, 720.0);
    let renderer = Engine::webgpu_renderer(&config)
        .await
        .map_err(|e| format!("webgpu_renderer init: {e:?}"))?;

    // Surface (render) pipeline. The vertex shader generates positions
    // procedurally from `vertex_index`, so we don't need any vertex
    // buffers — the pipeline layout is empty.
    let render_pipeline = renderer.create_render_pipeline_full(
        SURFACE_SHADER,
        &[],
        "vs_main",
        "fs_main",
        None,
    );
    if render_pipeline.is_undefined() || render_pipeline.is_null() {
        return Err(JsValue::from_str("create_render_pipeline_full returned undefined"));
    }

    // Height-storage buffer — created via euv-engine so it lives on the same
    // device the bind group / pipeline were created from. The vertex
    // shader reads it as `var<storage, read> array<f32>`.
    let grid_count = (GRID_RESOLUTION * GRID_RESOLUTION) as usize;
    let height_buffer = renderer.create_uniform_buffer(&vec![0.0_f32; grid_count]);
    if height_buffer.is_undefined() || height_buffer.is_null() {
        return Err(JsValue::from_str("create_uniform_buffer (heights) failed"));
    }

    // The bind group layout must list a Buffer entry for `@binding(1)`. The
    // height-buffer entry covers binding 1 (storage/read) and the surface
    // uniform covers binding 0.
    let surface_uniform = renderer.create_uniform_buffer(&build_surface_uniforms(
        &[0.0; 16],
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        0.0,
        (0.0, 0.0),
        0.0,
    ));

    let entries = vec![
        BindGroupEntry::Buffer {
            binding: 0,
            buffer: surface_uniform.clone(),
            offset: 0,
            size: None,
        },
        BindGroupEntry::Buffer {
            binding: 1,
            buffer: height_buffer.clone(),
            offset: 0,
            size: None,
        },
    ];
    let surface_bind_group = renderer.create_bind_group(&render_pipeline, 0, &entries);
    if surface_bind_group.is_undefined() || surface_bind_group.is_null() {
        return Err(JsValue::from_str("create_bind_group (surface) failed"));
    }

    let vertex_buffer = renderer.create_vertex_buffer(&build_grid_vertex_buffer());
    let index_buffer = renderer.create_index_buffer(&build_grid_index_buffer());
    let vertex_count = ((GRID_RESOLUTION as u32 - 1) * (GRID_RESOLUTION as u32 - 1) * 6) as u32;

    state.get_ready().set(true);

    Ok(WaterRenderer {
        renderer,
        render_pipeline,
        height_buffer,
        surface_uniform,
        surface_bind_group,
        vertex_buffer,
        index_buffer,
        vertex_count,
        mesh_origin: Rc::new(RefCell::new((0.0, 0.0))),
        sim: Rc::new(RefCell::new(WaveSimulation::new())),
        orbit,
        cancelled: Rc::new(Cell::new(false)),
        raf_id: Rc::new(Cell::new(None)),
        state,
    })
}


async fn run_water_loop(renderer: Rc<RefCell<WaterRenderer>>) {
    let start_time = current_time_ms();
    let mut last_frame = start_time;
    let mut frame_count: u32 = 0;
    let mut ambient_accum: f32 = 0.0;

    // Seed a few ripples for visual interest on first load.
    {
        let mut sim_ref = renderer.borrow();
        let mut sim = sim_ref.sim.borrow_mut();
        sim.inject_gaussian(GRID_RESOLUTION as i32 / 2, GRID_RESOLUTION as i32 / 2, 0.6, 8);
        sim.inject_gaussian(GRID_RESOLUTION as i32 / 2 + 16, GRID_RESOLUTION as i32 / 2 - 12, 0.4, 6);
        sim.inject_gaussian(GRID_RESOLUTION as i32 / 2 - 20, GRID_RESOLUTION as i32 / 2 + 18, 0.5, 7);
    }

    let renderer_for_closure = renderer.clone();
    let cancelled = renderer.borrow().cancelled.clone();
    let raf_id_cell = renderer.borrow().raf_id.clone();
    let raf_id_for_inner = raf_id_cell.clone();

    // Build the RAF closure and stash a clone of its `js_sys::Function`
    // handle so we can re-schedule itself each frame after it's moved
    // into the FnMut body.
    let raf_fn_cell: Rc<RefCell<Option<js_sys::Function>>> = Rc::new(RefCell::new(None));
    let raf_fn_cell_inner = raf_fn_cell.clone();
    let closure: Closure<dyn FnMut(f64)> = Closure::wrap(Box::new(move |_t: f64| {
        if cancelled.get() {
            return;
        }
        let now = current_time_ms();
        let dt = ((now - last_frame) as f32 / 1000.0).min(0.05);
        last_frame = now;
        let elapsed = (now - start_time) as f32 / 1000.0;
        frame_count += 1;
        if frame_count % 30 == 0 {
            let fps = 1.0 / dt.max(1e-3);
            renderer_for_closure.borrow().state.get_fps().set(fps);
        }

        // Step 1: ambient ripples.
        ambient_accum += dt;
        let ambient_interval = 1.0 / AMBIENT_RIPPLE_RATE_HZ;
        while ambient_accum >= ambient_interval {
            ambient_accum -= ambient_interval;
            let r = renderer_for_closure.borrow();
            let mut sim = r.sim.borrow_mut();
            let cx = (Math::random() as f32 * GRID_RESOLUTION as f32) as i32;
            let cy = (Math::random() as f32 * GRID_RESOLUTION as f32) as i32;
            sim.inject_gaussian(cx, cy, 0.25, 5);
        }

        // Step 2: integrate wave equation.
        {
            let r = renderer_for_closure.borrow();
            let mut sim = r.sim.borrow_mut();
            for _ in 0..2 {
                sim.step(WAVE_C_SQUARED, WAVE_DAMPING);
            }
        }

        // Step 3: upload the simulation buffer to the GPU height-storage
        // buffer via euv-engine's `update_uniform_buffer`.
        let curr_slice = {
            let r = renderer_for_closure.borrow();
            r.sim.borrow().curr.clone()
        };
        {
            let mut r = renderer_for_closure.borrow_mut();
            r.renderer
                .update_uniform_buffer(&r.height_buffer, &curr_slice);
        }

        // Step 4: snap mesh origin to the camera.
        renderer_for_closure.borrow().snap_mesh_to_camera();

        // Step 5: rebuild surface uniforms.
        let (cam, forward, mesh_origin) = {
            let r = renderer_for_closure.borrow();
            (
                r.orbit.camera_position(),
                camera_forward(&r.orbit),
                *r.mesh_origin.borrow(),
            )
        };
        let canvas_size = read_canvas_size(WATER_CANVAS_SELECTOR).unwrap_or((1280.0, 720.0));
        let view_proj = build_view_proj(cam, canvas_size.0, canvas_size.1);
        let uniforms = build_surface_uniforms(
            &view_proj,
            cam,
            forward,
            elapsed,
            mesh_origin,
            WaterRenderer::cell_size(),
        );
        {
            let r = renderer_for_closure.borrow();
            r.renderer
                .update_uniform_buffer(&r.surface_uniform, &uniforms);
        }

        // Step 6: render the surface via the high-level API.
        let render_pipeline = {
            renderer_for_closure.borrow().render_pipeline.clone()
        };
        let surface_bind_group = {
            renderer_for_closure.borrow().surface_bind_group.clone()
        };
        let vertex_count = renderer_for_closure.borrow().vertex_count;
        if frame_count == 1 {
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "[water] first render: bg={:?}, pipeline={:?}, vertex_count={}",
                if surface_bind_group.is_undefined() { "undef" } else { "ok" },
                if render_pipeline.is_undefined() { "undef" } else { "ok" },
                vertex_count,
            )));
        }
        let mut renderer_mut = renderer_for_closure.borrow_mut();
        renderer_mut.renderer.render_frame_with_bind_group(
            &render_pipeline,
            &surface_bind_group,
            (0.55, 0.72, 0.86, 1.0),
            vertex_count,
        );
        drop(renderer_mut);

        // Step 7: schedule next frame via the stashed Function handle.
        if let Some(win) = window() {
            let raf_fn = raf_fn_cell_inner
                .borrow()
                .as_ref()
                .expect("raf function")
                .clone();
            let raf_value = win.request_animation_frame(&raf_fn);
            if let Ok(id) = raf_value {
                raf_id_for_inner.set(Some(id));
            }
        }
    }));

    let raf_fn: js_sys::Function = closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
    *raf_fn_cell.borrow_mut() = Some(raf_fn);

    if let Some(win) = window() {
        let raf_fn = raf_fn_cell
            .borrow()
            .as_ref()
            .expect("raf function")
            .clone();
        let raf_value = win.request_animation_frame(&raf_fn);
        if let Ok(id) = raf_value {
            raf_id_cell.set(Some(id));
        }
    }

    closure.forget();
}




fn current_time_ms() -> f64 {
    js_sys::Date::now()
}
