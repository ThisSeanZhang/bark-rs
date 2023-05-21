use rust_bert::pipelines::masked_language::MaskedLanguageModel;
use tch::{Tensor, Device};

use crate::{bark_gpt::BarkGPT, model::fine::FineGPT, generation::generate_text_semantic};

const TEXT_ENCODING_OFFSET: i64 = 10_048;
const SEMANTIC_PAD_TOKEN: i64 = 10_000;
const TEXT_PAD_TOKEN: i64 = 129_595;
const SEMANTIC_INFER_TOKEN: i64 = 129_599;

// 生成语音
///
/// Args:
///     text: text to be turned into audio
/// 
///     history_prompt: history choice for audio cloning
/// 
///     text_temp: generation temperature (1.0 more diverse, 0.0 more conservative) default is 0.7
/// 
///     waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
/// 
///     silent: disable progress bar

///     output_full: return full generation to be used as a history prompt
/// 
/// Returns:
///     numpy audio array at sample frequency 24khz
pub fn generate_audio(
  models: BarkModel,
  text: &str,
  history_prompt: Option<String>,
  text_temp:f64, // 0.7,
  waveform_temp: f64, // float = 0.7,
  silent: bool, // False,
  output_full: bool //  False,
) -> Vec<u32> {
  let semantic_tokens = text_to_semantic(
    &models,
    text,
    history_prompt,
    text_temp, // 0.7,
    silent // False,
  );
  // let mut out = semantic_to_waveform(
  //     semantic_tokens,
  //     history_prompt=history_prompt,
  //     temp=waveform_temp,
  //     silent=silent,
  //     output_full=output_full,
  // );
  // if output_full {
  //   full_generation, audio_arr = out
  //   return full_generation, audio_arr
  // } else {
  //   audio_arr = out
  // }
  return vec![]
}

/// 文版转语义
/// 
/// Args:
///     text: text to be turned into audio
///     history_prompt: history choice for audio cloning
///     temp: generation temperature (1.0 more diverse, 0.0 more conservative)
///     silent: disable progress bar
/// 
/// Returns:
///    numpy semantic array to be fed into `semantic_to_waveform`
fn text_to_semantic(
  models: &BarkModel,
  text: &str,
  history_prompt: Option<String>,
  temp: f64, // 0.7,
  silent: bool // False,
) -> Tensor {

  let x_semantic = generate_text_semantic(
    models,
    text,
    history_prompt, // =None,
    temp, // =0.7,
    None, // =None,
    None, // =None,
    silent, // false
    0.2, //=0.2,
    None, //=None,
    true, // true,
    false, // False,
  );
  return x_semantic
}

pub struct BarkModel {
  pub text: BarkGPT,
  pub tokenizer: Tokenizer,
  // pub coarse: BarkGPT,
  // pub fine: FineGPT
  pub device: Device,
}

pub struct Tokenizer {
  pub inner: MaskedLanguageModel,
}

impl Tokenizer {
  pub fn tokenize(&self, text: &str) -> Vec<i64> {
    let tokens = self.inner.get_tokenizer().tokenize(text);
    self.inner.get_tokenizer().convert_tokens_to_ids(&tokens)
    // let tensor = Tensor::of_slice(&token_ids);
    // tensor + TEXT_ENCODING_OFFSET
  }
}

