use super::*;

use std::cell::Cell;
use std::rc::Rc;

use euv::Signal;
use lombok_macros::{Data, New};

#[derive(Clone, Data, New)]
pub(crate) struct UseWater {
    #[new(value = "euv::App::use_signal(|| 0.0_f32)")]
    #[get(pub(crate), type(copy))]
    pub(crate) fps: Signal<f32>,
    #[new(value = "euv::App::use_signal(|| false)")]
    #[get(pub(crate), type(copy))]
    pub(crate) ready: Signal<bool>,
    #[new(value = "euv::App::use_signal(String::new)")]
    #[get(pub(crate), type(copy))]
    pub(crate) error_message: Signal<String>,
}

#[derive(Clone)]
pub(crate) struct CameraOrbit {
    pub(crate) yaw: Rc<Cell<f32>>,
    pub(crate) pitch: Rc<Cell<f32>>,
    pub(crate) distance: Rc<Cell<f32>>,
}

impl CameraOrbit {
    pub(crate) fn new() -> CameraOrbit {
        CameraOrbit {
            yaw: Rc::new(Cell::new(INITIAL_YAW_RAD)),
            pitch: Rc::new(Cell::new(INITIAL_PITCH_RAD)),
            distance: Rc::new(Cell::new(INITIAL_DISTANCE_M)),
        }
    }

    pub(crate) fn add_yaw(&self, delta: f32) {
        self.yaw.set(self.yaw.get() + delta);
    }

    pub(crate) fn add_pitch(&self, delta: f32) {
        let next: f32 = (self.pitch.get() + delta).clamp(-1.3_f32, 1.3_f32);
        self.pitch.set(next);
    }

    pub(crate) fn multiply_distance(&self, factor: f32) {
        let next: f32 = (self.distance.get() * factor).clamp(MIN_DISTANCE_M, MAX_DISTANCE_M);
        self.distance.set(next);
    }

    pub(crate) fn camera_position(&self) -> (f32, f32, f32) {
        let yaw: f32 = self.yaw.get();
        let pitch: f32 = self.pitch.get();
        let distance: f32 = self.distance.get();
        let cy: f32 = pitch.cos();
        (
            distance * yaw.sin() * cy,
            distance * pitch.sin(),
            distance * yaw.cos() * cy,
        )
    }
}
