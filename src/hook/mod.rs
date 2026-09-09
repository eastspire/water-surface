//! Water simulation runtime: state, RAF loop, WebGPU pipelines, gestures.

mod r#const;
mod r#fn;
mod r#impl;
mod r#struct;

pub(crate) use r#fn::*;
pub(crate) use r#struct::*;

// Re-export const explicitly (pub(crate) items don't glob-reexport).
pub(crate) use r#const::{
    MAX_DISTANCE_M, MIN_DISTANCE_M, PINCH_SENSITIVITY,
    ROTATE_SENSITIVITY,
};
