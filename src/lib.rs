mod hook;
mod shader;
mod view;

#[allow(unused_imports)]
pub(crate) use {hook::*, shader::*, view::*};

use {euv::*, wasm_bindgen::prelude::*};

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    App::mount("#app", view::app);
}
