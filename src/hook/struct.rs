//! State structs and shared types.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use euv::Signal;
use lombok_macros::Getter;

/// Page-level reactive state for the water surface page.
///
/// All fields are `Signal`-wrapped so the view re-renders when any of them
/// change. The runtime signals (e.g. FPS) are updated from the RAF loop.
#[derive(Clone, Getter)]
pub(crate) struct UseWater {
    /// Current FPS estimate, smoothed.
    #[get(type(copy))]
    pub(crate) fps: Signal<f32>,
    /// True once the WebGPU pipelines have been created and the first
    /// frame has rendered successfully.
    #[get(type(copy))]
    pub(crate) ready: Signal<bool>,
    /// Last error message — empty string means no error.
    #[get(type(copy))]
    pub(crate) error_message: Signal<String>,
}

/// Mutable camera orbit state shared between the gesture handlers and the
/// RAF loop. Stored in `Rc<RefCell<...>>` so the gesture closures can update
/// it without going through the reactivity system (which would re-render the
/// page on every pointer move).
#[derive(Clone)]
pub(crate) struct CameraOrbit {
    pub(crate) yaw: Rc<Cell<f32>>,
    pub(crate) pitch: Rc<Cell<f32>>,
    pub(crate) distance: Rc<Cell<f32>>,
}

impl CameraOrbit {
    pub(crate) fn new() -> Self {
        use crate::hook::r#const::{
            INITIAL_DISTANCE_M, INITIAL_PITCH_RAD, INITIAL_YAW_RAD,
        };
        Self {
            yaw: Rc::new(Cell::new(INITIAL_YAW_RAD)),
            pitch: Rc::new(Cell::new(INITIAL_PITCH_RAD)),
            distance: Rc::new(Cell::new(INITIAL_DISTANCE_M)),
        }
    }

    pub(crate) fn add_yaw(&self, delta: f32) {
        self.yaw.set(self.yaw.get() + delta);
    }

    pub(crate) fn add_pitch(&self, delta: f32) {
        let next = (self.pitch.get() + delta).clamp(-1.3, 1.3);
        self.pitch.set(next);
    }

    pub(crate) fn multiply_distance(&self, factor: f32) {
        use crate::hook::r#const::{MAX_DISTANCE_M, MIN_DISTANCE_M};
        let next = (self.distance.get() * factor).clamp(MIN_DISTANCE_M, MAX_DISTANCE_M);
        self.distance.set(next);
    }

    /// Computes the world-space camera position from the orbit angles.
    pub(crate) fn camera_position(&self) -> (f32, f32, f32) {
        let yaw = self.yaw.get();
        let pitch = self.pitch.get();
        let distance = self.distance.get();
        let cy = pitch.cos();
        (
            distance * yaw.sin() * cy,
            distance * pitch.sin(),
            distance * yaw.cos() * cy,
        )
    }
}

impl Default for CameraOrbit {
    fn default() -> Self {
        Self::new()
    }
}
