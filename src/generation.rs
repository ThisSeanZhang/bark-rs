use std::{env, path::PathBuf, fmt, fs, borrow::Borrow};

use log::{warn, info};
use rust_bert::{pipelines::{masked_language::{MaskedLanguageModel, MaskedLanguageConfig}, common::ModelType}, resources::{LocalResource, RemoteResource}};
use tch::{Device, Cuda, nn::{Module, VarStore}, Tensor, IndexOp, Kind};
use indicatif::ProgressBar;

use crate::{bark_gpt::{Config, BarkGPT}, api::{BarkModel, Tokenizer}, model::{fine::FineGPT, self}};

const TEXT_ENCODING_OFFSET: i64 = 10_048;
const SEMANTIC_PAD_TOKEN: i64 = 10_000;
const TEXT_PAD_TOKEN: i64 = 129_595;
const TEXT_PAD_TOKEN_WITHOUT_OFFSET: i64 = 119_547;
const SEMANTIC_INFER_TOKEN: i64 = 129_599;

const SEMANTIC_VOCAB_SIZE: i64 = 10_000;

#[derive(Debug)]
pub enum BarkModelType {
  Text,
  Coarse,
  Fine
}

impl fmt::Display for BarkModelType {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      match self {
        BarkModelType::Text => write!(f, "text"),
        BarkModelType::Coarse => write!(f, "coarse"),
        BarkModelType::Fine => write!(f, "fine"),
      }
  }
}

struct  ModelConfig {
  repo_id: String,
  file_name: String
}
impl ModelConfig {
  // remote model path
  fn get_config(use_small: bool, model_type: &BarkModelType) -> Self {
    match (model_type, use_small) {
      (BarkModelType::Text, true) => ModelConfig {
        repo_id: "suno/bark".to_string(),
        file_name: "text.safetensors".to_string()
      },
      (BarkModelType::Text, false) => ModelConfig {
        repo_id: "suno/bark".to_string(),
        file_name: "text_2.pt".to_string()
      },
      (BarkModelType::Coarse, true) => ModelConfig {
        repo_id: "suno/bark".to_string(),
        file_name: "coarse.safetensors".to_string()
      },
      (BarkModelType::Coarse, false) => ModelConfig {
        repo_id: "suno/bark".to_string(),
        file_name: "coarse_2.pt".to_string()
      },
      (BarkModelType::Fine, true) => ModelConfig {
        repo_id: "suno/bark".to_string(),
        file_name: "fine.safetensors".to_string()
      },
      (BarkModelType::Fine, false) => ModelConfig {
        repo_id: "suno/bark".to_string(),
        file_name: "fine_2.pt".to_string()
      },
    }
  }
}

fn grab_best_device(use_gpu: bool) -> Device {
  let global_enable_mps = env::var("SUNO_ENABLE_MPS").map(|value| value.parse::<bool>().unwrap_or(false)).unwrap_or(false);
  if Cuda::device_count() > 0 && use_gpu {
    Device::cuda_if_available()
  } else if tch::utils::has_mps() && use_gpu && global_enable_mps{
    Device::Mps
  } else {
    warn!("No GPU being used. Careful, inference might be very slow!");
    Device::Cpu
  }
}

pub fn _load_model(device: Device, use_small: bool) -> BarkModel {
  // match model_type {
  //   BarkModelType::Text => todo!(),
  //   BarkModelType::Coarse => todo!(),
  //   BarkModelType::Fine => todo!(),
  // }
  // remote model path
  // let config = ModelConfig::get_config(use_small, &model_type);


  let text_ckpt_path = _get_ckpt_path(&BarkModelType::Text, use_small);
  let coarse_ckpt_path = _get_ckpt_path(&BarkModelType::Coarse, use_small);
  let fine_ckpt_path = _get_ckpt_path(&BarkModelType::Fine, use_small);

  info!("device: {:?}", device);

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

  let coarse_config  = Config {
    n_layer: 12,
    n_head: 12,
    n_embd: 768,
    block_size: 1024,
    bias: false,
    input_vocab_size: 12096,
    output_vocab_size: 12096,
    dropout: 0.0,
    n_codes_total: 8,
    n_codes_given: 1,
  };

  let coarse_vs = VarStore::new(device);
  let coarse_path = coarse_vs.root() / "_orig_mod";
  let coarse = BarkGPT::new(coarse_path, coarse_config);

  // let result = vs.load(ckpt_path);
  let result = coarse_vs.read_safetensors(coarse_ckpt_path);
  // info!("vs: {:?}",vs);
  info!("==== course load result: {}", result.is_ok());
  let a = coarse_vs.variables();
  for (name, tensor) in a {
    info!("{name}, {:?}", tensor.size())
  }
  info!("====");

  // let fine_config  = Config {
  //   n_layer: 12,
  //   n_head: 12,
  //   n_embd: 768,
  //   block_size: 1024,
  //   bias: false,
  //   input_vocab_size: 1056,
  //   output_vocab_size: 1056,
  //   dropout: 0.0,
  //   n_codes_total: 8,
  //   n_codes_given: 1,
  // };

  // info!("fine config: {fine_config:?}");

  // let fine_vs = VarStore::new(device);
  // let fine_path = fine_vs.root() / "_orig_mod";
  // let fine = FineGPT::new(fine_path, fine_config);

  // let result = vs.load(ckpt_path);
  // let result = fine_vs.read_safetensors(fine_ckpt_path);
  // info!("vs: {:?}",vs);
  // info!("==== fine load result: {}", result.is_ok());
  // let a = fine_vs.variables();
  // for (name, tensor) in a {
  //   info!("{name}, {:?}", tensor.size())
  // }
  // info!("====");


  let tokenizer = load_tokenizer();
  BarkModel{
    text,
    // coarse,
    tokenizer,
    // fine,
    device,
  }

}

fn _clear_cuda_cache() {

  if tch::Cuda::is_available() {
    // torch.cuda.empty_cache()
    // torch.cuda.synchronize()
  }
}

pub fn load_tokenizer() -> Tokenizer {
  let sequence_classification_config = MaskedLanguageConfig::new(
    ModelType::Bert,
    LocalResource::from(get_cache_path().join("rust_model.ot")),
    RemoteResource::from_pretrained(("bert-base-multilingual-cased/config", "https://huggingface.co/bert-base-multilingual-cased/raw/main/config.json")),
    RemoteResource::from_pretrained(("bert-base-multilingual-cased/vocab", "https://huggingface.co/bert-base-multilingual-cased/raw/main/vocab.txt")),
    None,
    false,
    None,
    None,
    None,
  );
  let sequence_classification_model = MaskedLanguageModel::new(sequence_classification_config).unwrap();
  Tokenizer {
    inner: sequence_classification_model
  }
}

fn get_cache_path() -> PathBuf {
  env::current_dir().unwrap().clone().join("cache")
}

pub fn _get_ckpt_path(model_type: &BarkModelType, use_small: bool) -> PathBuf {
  let file_name = ModelConfig::get_config(use_small, model_type).file_name;
  // TODO: Maybe panic
  let path = get_cache_path().join(file_name);
  info!("using file {path:?}");
  if !path.exists() {
    info!("{model_type:?} model not found");
    panic!()
  }
  path
}

fn load_model(use_gpu: bool, use_small: bool, force_reload: bool) -> BarkModel {
  let device = grab_best_device(true);
  _clear_cuda_cache();
  let model = _load_model(device, use_small);
  model
}

pub struct PerloadConfig {
  pub text_use_gpu: bool,
  pub text_use_small: bool,
  pub coarse_use_gpu: bool,
  pub coarse_use_small: bool,
  pub codec_use_gpu: bool,
  pub fine_use_gpu: bool,
  pub fine_use_small: bool,
  pub force_reload: bool
}
impl Default for PerloadConfig {
    fn default() -> Self {
        Self { 
          text_use_gpu: true,
          text_use_small: false,
          coarse_use_gpu: true,
          coarse_use_small: false,
          codec_use_gpu: true,
          fine_use_gpu: false,
          fine_use_small: true,
          force_reload: false
        }
    }
}
pub fn preload_models(config: PerloadConfig) -> BarkModel {
  let _device = grab_best_device(true);
  let model = load_model(true, true, true);
  model
}

fn _load_history_prompt(history_prompt_input: String) -> Option<Tensor> {
  if history_prompt_input.ends_with(".npz") {
    let tensors  = Tensor::read_npz(history_prompt_input).unwrap();
    for (name, tensor) in tensors {
      if name == "semantic_prompt" {
        return Some(tensor)
      }
    }
    None
  } else {
    None
  }
}

/// 将文本语义化
pub fn generate_text_semantic(
  models: &BarkModel,
  text: &str,
  history_prompt: Option<String>, // =None,
  temp: f64, // =0.7,
  top_k: Option<i64>, // =None,
  top_p: Option<i64>, // =None,
  silent: bool, // false
  min_eos_p: f64, //=0.2,
  max_gen_duration_s: Option<usize>, //=None,
  allow_early_stop: bool, // true,
  use_kv_caching: bool, // False,
) -> Tensor {

  let text = normalize_whitespace(text);

  let semantic_history = if let Some(path) = history_prompt {
    _load_history_prompt(path)
  } else {
    None
  };

  let mut encoded_text = models.tokenizer.tokenize(&text);

  info!("encoded code is: {encoded_text:?}");
  if encoded_text.len() > 256 {
    let p = (encoded_text.len() - 256) as f64 / encoded_text.len() as f64 * 100.0;
    warn!(
        "warning, text too long, lopping of last {}%", 
        p.round()
    );
    encoded_text.truncate(256);
  } else if encoded_text.len() < 256 {
    encoded_text.resize(256, TEXT_PAD_TOKEN_WITHOUT_OFFSET);
  }

  let encoded_text = Tensor::of_slice(&encoded_text) + TEXT_ENCODING_OFFSET;
  info!("encoded code after add TEXT_ENCODING_OFFSET: {encoded_text:?}");
  info!("encoded code after add TEXT_ENCODING_OFFSET: {}", encoded_text.data());
  let semantic_history = if let Some(semantic_history) = semantic_history {
    semantic_history.to_kind(tch::Kind::Int64);
    // lop off if history is too long, pad if needed
    if semantic_history.size()[0] < 256 {
      semantic_history.pad(&[256], "constant", Some(SEMANTIC_PAD_TOKEN as f64));
    }
    semantic_history
  } else {
    Tensor::full(&[256], SEMANTIC_PAD_TOKEN, (tch::Kind::Int64, Device::Cpu))
  };

  let x = Tensor::cat(&[
    encoded_text, semantic_history, Tensor::of_slice(&[SEMANTIC_INFER_TOKEN])
  ], -1).unsqueeze(0);

  info!("input x shape: {:?}", x.size());
  let use_device = models.device.clone();

  let out = tch::no_grad(move|| -> Tensor {
    let mut x = x.to(use_device);
    // TODO: modify back
    let n_tot_steps = 768;
    let mut kv_cache: Option<Vec<Option<(Tensor, Tensor)>>> = None;

    
    let pb = ProgressBar::new(n_tot_steps);
    for index in 0..n_tot_steps {
      pb.inc(1);
      match index {
        753..=756 => {
          info!("x content: {:?}", x.data());
          info!("x content: {}", x.data());
        },
        _=> ()
      }
      // info!("index is: {index}");
      let input_x = if kv_cache.is_some() {
        x.i((.., -1))
      } else {
        x.i(..)
      };
      let (logits, new_kv_cache) = models.text.forward_t(index,&input_x , false, true, kv_cache, None, use_kv_caching);
      kv_cache = new_kv_cache;

      // must be [1, 1, 10048]
      // info!("logits shape: {:?}", logits.size());
      let mut relevant_logits = if allow_early_stop {
        Tensor::cat(&[
          logits.i((0, 0, ..SEMANTIC_VOCAB_SIZE)),
          logits.i((0, 0, SEMANTIC_PAD_TOKEN)).unsqueeze(0), // EOS
        ], -1)
      } else {
        logits.i((0, 0, ..SEMANTIC_VOCAB_SIZE))
      };
     
      if let Some(p) = top_p {
        let mut logits = relevant_logits.copy();
        let repalce_value_mask =  logits.argsort(-1, true)
            .softmax(-1, Kind::Float)
            .cumsum(1, Kind::Float)
            .gt(p);
        relevant_logits = logits.f_masked_fill_(&repalce_value_mask, -f64::INFINITY).unwrap();
      }

      // 没用到 先不折腾了
      // if let Some(k) = top_k {
      //   let repalce_value_mask =  relevant_logits.topk(k, dim, largest, sorted)
      //       .softmax(-1, Kind::Float)
      //       .cumsum(1, Kind::Float)
      //       .gt(p);
      // }

      let props = (relevant_logits / temp).softmax(-1, Kind::Float);
      // println!("props: {}", props.data());
      let item_next = props.multinomial(1, false);
      // println!("item_next.squeeze(): {}", item_next.squeeze());
      // println!("item_next.int64_value(&[0]): {}", item_next.int64_value(&[0]));
      if allow_early_stop 
        && (item_next.int64_value(&[0]) == SEMANTIC_VOCAB_SIZE
            || props.double_value(&[-1]) >= min_eos_p
        ) {
          pb.finish_with_message("early stop");
          // eos found, so break
          break;
      }
      // info!("x shape is: {:?}", x.size());
      // info!("item_next shape is: {:?}", item_next.size());
      x = Tensor::cat(&[
        x, item_next.unsqueeze(0)
      ], 1);
    }
    pb.finish_with_message("done");
    info!("after x shape is: {:?}", x.size());
    let out  = x.detach().squeeze().i(256 + 256 + 1 ..);
    info!("outshape is: {:?}", out.size());
    out
  });
  out
}

fn normalize_whitespace(text: &str) -> String {
  text.replace(r"\s+", " ").trim().to_owned()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rust_bert::{pipelines::{masked_language::{MaskedLanguageConfig, MaskedLanguageModel}, common::ModelType}, resources::{LocalResource, RemoteResource}};
    use tch::{Tensor, Kind};

    use crate::generation::get_cache_path;

  #[test]
  fn test_top_n() {
    let mut logits = Tensor::of_slice(&[3.0, 1.0, 6.0, 4.0, 5.0, 2.0]);
    logits = logits.reshape(&[2,3]);
    
    println!("{}", logits.data());
    // Top-p 截断 
    let sorted_indices = logits.copy().argsort(-1, true);

    println!("{}", sorted_indices.data());

    // 使用cumsum计算累积概率
    let cumulative_probs = sorted_indices.softmax(-1, Kind::Float).cumsum(1, Kind::Float);
    
    println!("{}", cumulative_probs.data());
    let top_p = 0.7;
    let top_p_indices = cumulative_probs.gt(top_p).totype(Kind::Int64);
    println!("gt: {}", top_p_indices.data());
    // let top_p_indices = top_p_indices.nonzero();
    // println!("nonzero: {}", top_p_indices.data());
    // let top_p_indices = top_p_indices.flatten(start_dim, end_dim)
    // println!("flat_view: {}", top_p_indices.data());
    // let aa = top_p_indices * -f64::INFINITY;
    
    // println!("{}", aa.data());
     let result = logits.f_masked_fill_(&top_p_indices, -f64::INFINITY).unwrap();
    println!("{}", result.data());

  }

  #[test]
  fn test_top_k() {
    let mut logits = Tensor::of_slice(&[3.0, 1.0, 6.0, 4.0, 5.0, 2.0]);
    logits = logits.reshape(&[2,3]);

    let k = 2;
    let (result1, result2) = logits.topk(k, -1, true, true);
    println!("{}", result1.data());
    println!("{}", result2.data());

  }
  #[test]
  fn test_token() {
    let sequence_classification_config = MaskedLanguageConfig::new(
      ModelType::Bert,
      LocalResource::from(get_cache_path().join("rust_model.ot")),
      RemoteResource::from_pretrained(("bert-base-multilingual-cased/config", "https://huggingface.co/bert-base-multilingual-cased/raw/main/config.json")),
      RemoteResource::from_pretrained(("bert-base-multilingual-cased/vocab", "https://huggingface.co/bert-base-multilingual-cased/raw/main/vocab.txt")),
      None,
      false,
      None,
      None,
      None,
  );

  let sequence_classification_model = MaskedLanguageModel::new(sequence_classification_config).unwrap();

      let input = [
          "Replace me by any text you'd like.",
      ];
  
      let c = sequence_classification_model.get_tokenizer().tokenize("Replace me by any text you'd like.");
      println!("{c:?}");
      let d = sequence_classification_model.get_tokenizer().convert_tokens_to_ids(&c);
      println!("{d:?}");
      let e = Tensor::of_slice(&d);
      println!("{}", e.data());
      let f = e + 10_048;
      println!("{}", f.data());
      // let a = sequence_classification_model.get_tokenizer().encode_list(&input, 256, &TruncationStrategy::DoNotTruncate, 0);
      // for each in a.iter() {
      //     // let b = each.token_ids
      //     println!("{each:?}");
      // }
      //    Run model
      let output = sequence_classification_model.predict(input);
      if let Ok(label) = output {
          println!("{label:?}");
      }
  }
}


