//! Water simulation runtime: state, RAF loop, WebGPU pipelines, gestures.

mod r#const;
mod r#fn;
mod r#impl;
mod r#struct;

pub(crate) use r#fn::*;
pub(crate) use r#struct::*;

// Re-export const explicitly (pub(crate) items don't glob-reexport).
pub(crate) use r#const::{
    AMBIENT_RIPPLE_RATE_HZ, FRESNEL_POWER, FRESNEL_REFLECTIVITY, GRID_RESOLUTION,
    HEIGHT_AMPLITUDE_M, INITIAL_DISTANCE_M, INITIAL_PITCH_RAD, INITIAL_YAW_RAD,
    MAX_DISTANCE_M, MESH_TILE_SIZE_M, MIN_DISTANCE_M, PINCH_SENSITIVITY,
    ROTATE_SENSITIVITY, SKY_HORIZON_COLOR, SKY_TOP_COLOR, SUN_COLOR, SUN_DIR_X,
    SUN_DIR_Y, SUN_DIR_Z, WATER_CANVAS_SELECTOR, WATER_COLOR, WAVE_C_SQUARED,
    WAVE_DAMPING,
};
