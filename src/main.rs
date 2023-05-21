use std::env;

use api::{generate_audio, BarkModel};
use bark_gpt::{Config, BarkGPT};
use generation::{generate_text_semantic, _load_model, _get_ckpt_path, BarkModelType, load_tokenizer};
use log::info;
use tch::{Device, nn::VarStore};

use crate::generation::{preload_models, PerloadConfig};

mod api;
mod model;
mod generation;
mod bark_gpt;

fn main() {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let bark_model = get_model(Device::cuda_if_available(), true);
    let text = " \
        Hello, my name is Suno. And, uh — and I like pizza. [laughs] \
        But I also have other interests such as playing tic tac toe. \
    ";
    let x_semantic = generate_text_semantic(
        &bark_model,
        text,
        None, // =None,
        0.7, // =0.7,
        None, // =None,
        None, // =None,
        false, // false
        0.2, //=0.2,
        None, //=None,
        true, // true,
        false, // False,
      );
}

fn get_model(device: Device, use_small: bool) -> BarkModel {
    let text_ckpt_path = _get_ckpt_path(&BarkModelType::Text, use_small);
    let text_config  = Config {
        n_layer: 12,
        n_head: 12,
        n_embd: 768,
        block_size: 1024,
        bias: false,
        input_vocab_size: 129600,
        output_vocab_size: 10048,
        dropout: 0.0,
        n_codes_total: 8,
        n_codes_given: 1,
      };
    let text_vs = VarStore::new(device);
    let text_path = text_vs.root() / "_orig_mod";
    let text = BarkGPT::new(text_path, text_config);

    let result = text_vs.read_safetensors(text_ckpt_path);
    info!("==== text load result: {}", result.is_ok());
    let a = text_vs.variables();
    for (name, tensor) in a {
        info!("{name}, {:?}", tensor.size())
    }
    info!("====");

    let tokenizer = load_tokenizer();
    BarkModel{
      text,
      // coarse,
      tokenizer,
      // fine,
      device,
    }
}

