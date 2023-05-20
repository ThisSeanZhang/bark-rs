use std::env;

use api::generate_audio;

use crate::generation::{preload_models, PerloadConfig};

mod api;
mod model;
mod generation;
mod bark_gpt;

fn main() {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let bark_model = preload_models(PerloadConfig::default());
    let text = " \
        Hello, my name is Suno. And, uh — and I like pizza. [laughs] \
        But I also have other interests such as playing tic tac toe. \
    ";
    let a = generate_audio(bark_model, text, None, 0.7, 0.7, false, false);
}
