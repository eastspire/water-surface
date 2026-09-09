mod hook;
mod shader;
mod view;

use euv::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    App::mount("#app", view::app);
}
