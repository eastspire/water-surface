//! Tunable constants for the water simulation and camera.

/// CSS selector for the canvas element the water is rendered into.
pub(crate) const WATER_CANVAS_SELECTOR: &str = "#water-canvas";

/// Square grid resolution for the wave simulation + mesh.
/// 256×256 = 65k vertices — small enough to integrate at 60 Hz, large
/// enough that the surface looks continuous at all camera distances.
pub(crate) const GRID_RESOLUTION: u32 = 256;

/// World-space size of one mesh tile in meters.
/// The tile follows the camera in world space, so the lake is effectively
/// infinite while only this much geometry is allocated.
pub(crate) const MESH_TILE_SIZE_M: f32 = 80.0;

/// Wave-equation tuning.
///
/// `c_squared` is the squared wave speed — controls how fast ripples travel.
/// `damping` is per-step velocity loss — controls how long ripples last.
pub(crate) const WAVE_C_SQUARED: f32 = 0.18;
pub(crate) const WAVE_DAMPING: f32 = 0.012;

/// Maximum displacement in meters. Larger amplitude gives bigger visible
/// peaks but exaggerates the height-vs-horizontal ratio.
pub(crate) const HEIGHT_AMPLITUDE_M: f32 = 1.4;


/// Initial camera orbit angles in radians (yaw, pitch, distance).
pub(crate) const INITIAL_YAW_RAD: f32 = 0.6;
pub(crate) const INITIAL_PITCH_RAD: f32 = 0.35;
pub(crate) const INITIAL_DISTANCE_M: f32 = 35.0;

/// Min / max camera distance in meters (pinch zoom bounds).
pub(crate) const MIN_DISTANCE_M: f32 = 4.0;
pub(crate) const MAX_DISTANCE_M: f32 = 220.0;

/// Sensitivity multipliers for the gesture handlers.
pub(crate) const ROTATE_SENSITIVITY: f32 = 0.005;
pub(crate) const PINCH_SENSITIVITY: f32 = 0.04;

/// Ambient ripple impulses (random taps on the lake surface) emitted per
/// second by the demo. Gives the lake its baseline motion.
pub(crate) const AMBIENT_RIPPLE_RATE_HZ: f32 = 1.4;

/// Sun direction — high in the sky, slightly behind the camera on first load.
pub(crate) const SUN_DIR_X: f32 = 0.45;
pub(crate) const SUN_DIR_Y: f32 = 0.78;
pub(crate) const SUN_DIR_Z: f32 = -0.42;

/// Water + sky palette (linear RGB).
pub(crate) const WATER_COLOR: (f32, f32, f32) = (0.04, 0.10, 0.16);
pub(crate) const SUN_COLOR: (f32, f32, f32) = (1.0, 0.94, 0.82);
pub(crate) const SKY_TOP_COLOR: (f32, f32, f32) = (0.18, 0.36, 0.66);
pub(crate) const SKY_HORIZON_COLOR: (f32, f32, f32) = (0.85, 0.86, 0.88);

/// Fresnel tuning. Lower `reflectivity` = darker water at normal incidence
/// (looking straight down) so deep-water color dominates near the camera;
/// higher `fresnel_power` = sharper reflection-vs-refraction transition at
/// grazing angles.
pub(crate) const FRESNEL_REFLECTIVITY: f32 = 0.04;
pub(crate) const FRESNEL_POWER: f32 = 5.0;
