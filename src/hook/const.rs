#[allow(unused_imports)]
use super::*;

pub(crate) const WATER_CANVAS_SELECTOR: &str = "#water-canvas";

pub(crate) const GRID_RESOLUTION: u32 = 256;

pub(crate) const MESH_TILE_SIZE_M: f32 = 80.0;

pub(crate) const WAVE_C_SQUARED: f32 = 0.18;
pub(crate) const WAVE_DAMPING: f32 = 0.012;

pub(crate) const HEIGHT_AMPLITUDE_M: f32 = 1.4;

pub(crate) const INITIAL_YAW_RAD: f32 = 0.6;
pub(crate) const INITIAL_PITCH_RAD: f32 = 0.35;
pub(crate) const INITIAL_DISTANCE_M: f32 = 35.0;

pub(crate) const MIN_DISTANCE_M: f32 = 4.0;
pub(crate) const MAX_DISTANCE_M: f32 = 220.0;

pub(crate) const ROTATE_SENSITIVITY: f32 = 0.005;
pub(crate) const PINCH_SENSITIVITY: f32 = 0.04;

pub(crate) const AMBIENT_RIPPLE_RATE_HZ: f32 = 1.4;

pub(crate) const SUN_DIR_X: f32 = 0.45;
pub(crate) const SUN_DIR_Y: f32 = 0.78;
pub(crate) const SUN_DIR_Z: f32 = -0.42;

pub(crate) const WATER_COLOR: (f32, f32, f32) = (0.04, 0.10, 0.16);
pub(crate) const SUN_COLOR: (f32, f32, f32) = (1.0, 0.94, 0.82);
pub(crate) const SKY_TOP_COLOR: (f32, f32, f32) = (0.18, 0.36, 0.66);
pub(crate) const SKY_HORIZON_COLOR: (f32, f32, f32) = (0.85, 0.86, 0.88);

pub(crate) const FRESNEL_REFLECTIVITY: f32 = 0.04;
pub(crate) const FRESNEL_POWER: f32 = 5.0;

pub(crate) const SURFACE_UNIFORM_F32_COUNT: usize = 52;
