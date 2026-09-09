use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum GestureState {
    #[default]
    Idle,
    Rotating {
        last_x: f32,
        last_y: f32,
    },
    Pinching {
        initial_distance: f32,
        initial_orbit_distance: f32,
    },
}
