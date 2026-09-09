//! Small trait impls.

use super::r#struct::{CameraOrbit, UseWater};

impl Default for UseWater {
    fn default() -> Self {
        Self {
            fps: euv::App::use_signal(|| 0.0_f32),
            ready: euv::App::use_signal(|| false),
            error_message: euv::App::use_signal(|| String::new()),
        }
    }
}

// `CameraOrbit::new` lives in struct.rs; this file exists to satisfy the
// 9-keyword-files rule for the hook/ directory.
#[allow(dead_code)]
fn _suppress_unused_warning() {
    let _ = CameraOrbit::new();
}
