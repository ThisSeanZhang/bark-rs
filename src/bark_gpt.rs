use std::borrow::Borrow;

use anyhow::{bail, Result};
use log::info;
use tch::data::TextData;
use tch::nn::{ModuleT, OptimizerConfig, LayerNormConfig, LayerNorm, LinearConfig, Sequential, Module};
use tch::{
  nn,
  Device,
  IndexOp,
  Kind,
  Tensor, Cuda
};

const LEARNING_RATE: f64 = 0.0003;
const BLOCK_SIZE: i64 = 128;
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 100;
const SAMPLING_LEN: i64 = 4096;

#[derive(Debug, Copy, Clone)]
pub struct Config {
  pub block_size: i64,
  pub input_vocab_size: i64,
  pub output_vocab_size: i64,
  pub n_layer: u64,
  pub n_head: i64,
  pub n_embd: i64,
  pub dropout: f64,
  pub bias: bool,
  // fine
  pub n_codes_total: i64,
  pub n_codes_given: i64,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      block_size: 1024,
      input_vocab_size: 10_048,
      output_vocab_size: 10_048,
      n_layer: 12,
      n_head: 12,
      n_embd: 768,
      dropout: 0_f64,
      bias: true,
      // fine
      n_codes_total:8,
      n_codes_given:1,
    }
  }
}
// Weight decay only applies to the weight matrixes in the linear layers
const NO_WEIGHT_DECAY_GROUP: usize = 0;
const WEIGHT_DECAY_GROUP: usize = 1;

// Custom linear layer so that different groups can be used for weight and biases.
#[derive(Debug)]
struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

impl nn::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
    }
}

// LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
// fn layer_norm<'a, T: std::borrow::Borrow<nn::Path<'a>>>(
//   vs: T,
//   normalized_shape: Vec<i64>,
//   config: LayerNormConfig,
//   has_bias: bool
// ) -> LayerNorm {
//   let vs = vs.borrow();

//     let (ws, bs) = if config.elementwise_affine {
//         let ws = Some(vs.var("weight", normalized_shape.as_slice(), config.ws_init));
//         let bs = if has_bias {
//           Some(vs.var("bias", normalized_shape.as_slice(), config.bs_init))
//         } else { None };
//         (ws, bs)
//     } else {
//         (None, None)
//     };

//     LayerNorm { config, ws, bs, normalized_shape }
// }

#[derive(Debug)]
pub struct MyLayerNorm {
    config: LayerNormConfig,
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
    pub normalized_shape: Vec<i64>,
}

pub fn layer_norm<'a, T: Borrow<nn::Path<'a>>>(
    vs: T,
    ndim: i64,
    has_bias: bool
) -> MyLayerNorm {
    let vs = vs.borrow();
    let normalized_shape = vec![ndim];
    let config = LayerNormConfig::default();
    let (ws, bs) = if config.elementwise_affine {
        let ws = Some(vs.var("weight", normalized_shape.as_slice(), config.ws_init));
        let bs = if has_bias {
          Some(vs.var("bias", normalized_shape.as_slice(), config.bs_init))
        } else { None };
        (ws, bs)
    } else {
        (None, None)
    };

    MyLayerNorm { config, ws, bs, normalized_shape }
}

impl nn::Module for MyLayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::layer_norm(
            xs,
            self.normalized_shape.as_slice(),
            self.ws.as_ref(),
            self.bs.as_ref(),
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}


  // // key, query, value projections for all heads, but in a batch
  // let c_attn = nn::linear(p / "c_attn", config.n_embd, config.n_embd * 3, LinearConfig{
  //   bias: config.bias,
  //   ..LinearConfig::default()
  // });
  // //output projection
  // let c_proj = nn::linear(p / "c_proj", config.n_embd, config.n_embd, LinearConfig{
  //   bias: config.bias,
  //   ..LinearConfig::default()
  // });

  #[derive(Debug)]
struct CausalSelfAttention {
  // key: nn::Linear,
  // query: nn::Linear,
  // value: nn::Linear,
  c_attn: nn::Linear,
  proj: nn::Linear,
  mask: Tensor, 

  n_head: i64,
  n_embd: i64,
  dropout: f64,
}
  
  impl CausalSelfAttention {
    fn new(p: &nn::Path, config: &Config) -> Self {
      // TODO: 合并
        // let key = nn::linear(p / "key", config.n_embd, config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() });
        // let query = nn::linear(p / "query", config.n_embd, config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() });
        // let value = nn::linear(p / "value", config.n_embd, config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() });

      let c_attn = nn::linear(p / "c_attn", config.n_embd, 3 * config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() });

      let proj = nn::linear(p / "c_proj", config.n_embd, config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() });
      
      let mask_init = Tensor::ones(&[config.block_size, config.block_size], (Kind::Float, p.device())).tril(0);
      let mask = mask_init.view([1, 1, config.block_size, config.block_size]);
      
      Self { c_attn, proj, mask,
        n_head: config.n_head,
        n_embd: config.n_embd,
        dropout: config.dropout,
      }
    } 
  }
  
  impl CausalSelfAttention {
    fn forward_t(&self, xs: &Tensor, train: bool,mut past_kv: Option<(Tensor, Tensor)>, use_cache: bool) -> (Tensor, Option<(Tensor, Tensor)>) {

      let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
    let sizes = [sz_b, sz_t, self.n_head, sz_c / self.n_head];

    // not sure: q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
    let kqv = xs.apply(&self.c_attn).chunk(3, 2);

    let mut k = kqv[0].view(sizes).transpose(1, 2); // (B, nh, T, hs)
    let q = kqv[1].view(sizes).transpose(1, 2); // (B, nh, T, hs)
    let mut v = kqv[2].view(sizes).transpose(1, 2); // (B, nh, T, hs)
    // info!("CausalSelfAttention k shape: {:?}", k.size());
    // info!("CausalSelfAttention q shape: {:?}", q.size());
    // info!("CausalSelfAttention v shape: {:?}", v.size());
    // let is_causal = past_kv.is_none();
    if let Some((past_key, past_value)) = past_kv.take() {
      k = Tensor::cat(&[past_key, k], -2);
      v = Tensor::cat(&[past_value, v], -2);
    }
    // let mask_init = Tensor::ones(&[sz_t, sz_t], (Kind::Float, xs.device())).tril(0);
    // let mask = mask_init.view([1, 1, sz_t, sz_t]);

    let att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));
    let att = att.masked_fill(&&self.mask.i((.., .., ..sz_t, ..sz_t)).eq(0.), std::f64::NEG_INFINITY);
    let att = att.softmax(-1, Kind::Float).dropout(0.0, train);
    let y = att.matmul(&v).transpose(1, 2).contiguous().view([sz_b, sz_t, sz_c]);
    
    // let mask = if is_causal {
    //   None
    // } else {
    //   Some(&mask)
    // };
    // let y = tch::Tensor::scaled_dot_product_attention(&q, &k, &v, mask, self.dropout, is_causal);
    let ys = y.transpose(1, 2).contiguous().view([sz_b, sz_t, sz_c]);

    // TODO: reflash cache
    if use_cache {
      past_kv = Some((k, v));
    }
    let result = self.proj.forward_t(&ys, train).dropout(self.dropout, train);
    (result, past_kv)
    }
  }

// fn causal_self_attention(p: &nn::Path, config: Config, past_kv: Option<(Tensor, Tensor)>, use_cache: bool) -> impl ModuleT {
//   assert!(config.n_embd % config.n_head == 0);

//   let key = nn::linear(p / "key", config.n_embd, config.n_embd, LinearConfig{
//     bias: config.bias,
//     ..LinearConfig::default()
//   });
//   let query = nn::linear(p / "query", config.n_embd, config.n_embd, LinearConfig{
//     bias: config.bias,
//     ..LinearConfig::default()
//   });
//   let value = nn::linear(p / "value", config.n_embd, config.n_embd, LinearConfig{
//     bias: config.bias,
//     ..LinearConfig::default()
//   });
//   let proj = nn::linear(p / "proj", config.n_embd, config.n_embd, LinearConfig{
//     bias: config.bias,
//     ..LinearConfig::default()
//   });

  
//   let mask_init = Tensor::ones(&[config.block_size, config.block_size], (Kind::Float, p.device())).tril(0);
//   let mask_init = mask_init.view([1, 1, config.block_size, config.block_size]);
//   // let mask = p.var_copy("mask", &mask_init);
//   let mask = mask_init;
//   nn::func_t(move |xs, train| {
//     // batch size, sequence length, embedding dimensionality (n_embd)
//     let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
//     let sizes = [sz_b, sz_t, config.n_head, sz_c / config.n_head];

//     let mut k = xs.apply(&key).view(sizes).transpose(1, 2);
//     let q = xs.apply(&query).view(sizes).transpose(1, 2);
//     let mut v = xs.apply(&value).view(sizes).transpose(1, 2);

//     if let Some((past_key, past_value)) = past_kv {
//       k = Tensor::cat(&[past_key, k], -2);
//       v = Tensor::cat(&[past_value, k], -2);
//     }

//     let att = q.matmul(&k.transpose(-2, -1)) * (1.0 / f64::sqrt(sizes[3] as f64));
//     let att = att.masked_fill(&mask.i((.., .., ..sz_t, ..sz_t)).eq(0.), std::f64::NEG_INFINITY);
//     let att = att.softmax(-1, Kind::Float).dropout(config.dropopt, train);
//     let ys = att.matmul(&v).transpose(1, 2).contiguous().view([sz_b, sz_t, sz_c]);
//     ys.apply(&proj).dropout(dropopt.resid_pdrop, train)
//   })
// }

#[derive(Debug)]
pub struct MLP {
  pub c_fc: nn::Linear,
  pub c_proj: nn::Linear,
  pub drop: f64
}

impl ModuleT for MLP {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
      let xs = self.c_fc.forward(&xs).gelu("none");
      let xs = self.c_proj.forward(&xs);
      xs.dropout(self.drop, train)
    }
}

// fn mlp(p: &nn::Path, config: &Config) -> impl ModuleT {
//   let c_fc = nn::linear(p / "c_fc", config.n_embd, 4 * config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() });
//   let c_proj = nn::linear(p / "c_proj", 4 * config.n_embd, config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() });
//   let drop = config.dropopt;
//   nn::func_t(move |xs, train| {
//     let xs = c_fc.forward(&xs);
//     let xs = c_proj.forward(&xs);
//     let xs = xs.dropout(drop, train);
//     xs.gelu("none")
//   })
// }

struct Block {
  ln_1: MyLayerNorm,
  attn: CausalSelfAttention,
  ln_2: MyLayerNorm,
  mlp: MLP,
  layer_idx: u64
}

impl Block {
  fn new(p: &nn::Path, config: &Config, layer_idx: u64) -> Self{
    let path = p / layer_idx;
    Self {
      ln_1: layer_norm(&path / "ln_1", config.n_embd, config.bias),
      ln_2: layer_norm(&path / "ln_2", config.n_embd, config.bias),
      layer_idx,
      attn: CausalSelfAttention::new(&(&path/ "attn"), config),
      mlp: MLP {
        c_fc: nn::linear(&path / "mlp" / "c_fc", config.n_embd, 4 * config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() }),
        c_proj: nn::linear(&path / "mlp" / "c_proj", 4 * config.n_embd, config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() }),
        drop: config.dropout,
      }
    }
  }

  fn forward_t(&self, xs: &Tensor, train: bool, past_kv: Option<(Tensor, Tensor)>, use_cache: bool) -> (Tensor, Option<(Tensor, Tensor)>) {
    let (attn_output, prev_kvs) = self.attn.forward_t(&self.ln_1.forward_t(&xs, train), train,  past_kv, use_cache);
    let x = xs + attn_output;
    let x = &x + self.mlp.forward_t(&self.ln_2.forward_t(&x, train), train);
    return (x, prev_kvs)
  }

}
pub struct BarkGPT {
  config: Config,
  wte: nn::Embedding,
  wpe: nn::Embedding,
  drop: f64,
  h: Vec<Block>,
  ln_f: MyLayerNorm,
  lm_head: nn::Linear,
}

impl BarkGPT {

  pub fn new(p: nn::Path, config: Config) -> Self {
    let transformer = &p / "transformer";
    let tok_emb = nn::embedding(&transformer / "wte", config.input_vocab_size, config.n_embd, Default::default());
    let pos_emb = nn::embedding(&transformer / "wpe", config.block_size, config.n_embd, Default::default());
    let drop = config.dropout;

    let mut blocks = vec![];
    for block_idx in 0..config.n_layer {
        blocks.push(Block::new(&(&transformer / "h"), &config, block_idx));
    }
    
    let ln_f = layer_norm(&transformer / "ln_f", config.n_embd, false);
    let lm_head = nn::linear(&p /  "lm_head", config.n_embd, config.output_vocab_size, LinearConfig { bias: false, ..LinearConfig::default() });
    
    // info!("ln_f shape {:?}", if ln_f.ws.is_some() { ln_f.ws.unwrap().size()} else { vec![] });
    // info!("tok_emb shape {:?}", tok_emb.ws.size());
    // info!("pos_emb shape {:?}", pos_emb.ws.size());
    // info!("lm_head shape {:?}", lm_head.ws.size());
    BarkGPT {
      config,
      wte: tok_emb,
      wpe: pos_emb,
      drop,
      h: blocks,
      ln_f,
      lm_head
    }
  }

  fn get_num_params(&self, non_embedding: bool) -> usize {
    let mut sum = 0;
    if !non_embedding {
      sum += self.wte.ws.numel();
      sum += self.wpe.ws.numel();
    }
    // sum += self.ln_f.ws.some;
    sum += self.lm_head.ws.numel();
    sum
  }

  // idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False
  pub fn forward_t(&self, index: u64, idx: &Tensor, train: bool, merge_context: bool ,past_kv: Option<Vec<Option<(Tensor, Tensor)>>>, use_position_ids: Option<Tensor>, use_cache: bool) -> (Tensor, Option<Vec<Option<(Tensor, Tensor)>>>) {
    // info!("input x shape: {:?}", idx.size());
    let device = idx.device();
    // info!("using device is : {:?}", device);
    let size = idx.size();
    let ( b, mut t) = (size[0], size[1]);
    info!("size is : {size:?}, t is: {t}");
    // let (b,mut t, _) = idx.size3().unwrap();
    // info!("b: {b:}, t: {t:}");
    // info!("input x size2: {b} {t}");
    let tok_emb = if past_kv.is_some() {
      assert!( t == 1);
      self.wte.forward_t(idx, train)
    } else {
      if merge_context {
        assert!(idx.size()[1] >= 256+256+1);
        t = idx.size()[1] - 256
      } else {
        assert!( t <= self.config.block_size, "Cannot forward sequence of length {t:}, block size is only {}", self.config.block_size);
      }
      // forward the GPT model itself
      if merge_context {
        let idx_1 = idx.i((.., ..256)).to_device(device);
        let idx_2 = idx.i((.., 256..512)).to_device(device);
        let idx_3 = idx.i((.., 512..)).to_device(device);

        let tok_emb_1 = self.wte.forward_t(&idx_1, train);
        let tok_emb_2 = self.wte.forward_t(&idx_2, train);
        let tok_emb_3 = self.wte.forward_t(&idx_3, train);

        Tensor::cat(&[tok_emb_1 + tok_emb_2, tok_emb_3], 1)
      } else {
        self.wte.forward_t(&idx, train) // token embeddings of shape (b, t, n_embd)
      }
    };

    
    // info!("tok_emb shape: {}", tok_emb);
    // info!("tok_emb shape: {:?}", tok_emb.size());

    let (past_length, use_past_kv) = if let Some(past) = past_kv {
      let size = past[0].as_ref().unwrap().0.size();
      let past_length = size[size.len() - 2];
      // [1, 12, 257, 64]
      // past_length = 257
      info!("past_length is: {past_length}");
      (past_length, past) 
    } else {
      let mut reuslt: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(self.h.len());
      for i in 0..self.h.len() {
        reuslt.push(None);
      }
      (0, reuslt)
    };

    let position_ids = if let Some(position_ids) = use_position_ids {
      position_ids
    } else {
      // https://pytorch.org/docs/stable/generated/torch.Tensor.long.html
      let mut position_ids = Tensor::arange_start(past_length, t + past_length, (Kind::Int64, idx.device()));
      position_ids = position_ids.unsqueeze(0);
      assert_eq!(position_ids.size(), &[1, t]);
      position_ids
    };
    
    match index {
      0 | 1 | 2 | 753..=756 => {
        info!("index is: {index}, t is: {t}");
        info!("position_ids shape: {:?}", position_ids.data());
        info!("position_ids content: {}", position_ids);
        info!("position_ids max: {}", position_ids.max());
        info!("wpe shape: {:?}", self.wpe.ws.data());
      },
      _=> ()
    }
    
    let pos_emb = self.wpe.forward(&position_ids).to_device(device);
    match index {
      753..=756 => {
        info!("pos_emb shape: {:?}", pos_emb.data());
      },
      _=> ()
    }

    let x = (tok_emb + pos_emb).dropout(self.drop, train);
    
    // info!("before block x shape: {:?}", x.size());
    
    let (result, cache) = if use_cache {
      let mut new_kv = vec![];
      let mut run_x = x;
      for (block, past_layer_kv) in self.h.iter().zip(use_past_kv.into_iter()) {
          let (x, kv) = block.forward_t(&run_x, train, past_layer_kv, true);
          run_x = x;
          new_kv.push(kv);
      }
      (run_x, Some(new_kv))
    } else {
      let mut run_x = x;
        for block in self.h.iter() {
          let (x, _) = block.forward_t(&run_x, train, None, use_cache);
          run_x = x;
            // let (x, _) = block.forward_t(&x, train, None, false);
        }
        (run_x, None)
    };
    
    let result = self.ln_f.forward_t(&result, train);
    // info!("result: {}", result);
    let a = result.i((.., -1, ..)).unsqueeze(0);
    // info!("result.i((.., -1, ..)) data: {}", a);
    // info!("result.i((.., -1, ..)) shape: {:?}", a.size());
    // info!("result shape: {:?}", result.select(1, -1));
    let result = self.lm_head.forward_t(&a, train);
    // info!("result shape: {:?}", result.size());
    (result, cache)
    
  }
}




