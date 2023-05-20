use log::info;
use tch::{Tensor, nn::{self, LinearConfig, ModuleT}, Kind, index, IndexOp};

use crate::bark_gpt::{Config, MyLayerNorm, MLP, layer_norm};


struct NonCausalSelfAttention {
  c_attn: nn::Linear,
  proj: nn::Linear,
  mask: Tensor, 

  n_head: i64,
  n_embd: i64,
  dropout: f64,
}

impl NonCausalSelfAttention {

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

  fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {

    let (sz_b, sz_t, sz_c) = xs.size3().unwrap();
    let sizes = [sz_b, sz_t, self.n_head, sz_c / self.n_head];

    // not sure: q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
    let kqv = xs.apply(&self.c_attn).chunk(3, 2);

    let k = kqv[0].view(sizes).transpose(1, 2); // (B, nh, T, hs)
    let q = kqv[1].view(sizes).transpose(1, 2); // (B, nh, T, hs)
    let v = kqv[2].view(sizes).transpose(1, 2); // (B, nh, T, hs)

    let y = tch::Tensor::scaled_dot_product_attention(&q, &k, &v, Some(&self.mask), self.dropout, false);
    let ys = y.transpose(1, 2).contiguous().view([sz_b, sz_t, sz_c]);

    ys.apply(&self.proj).dropout(self.dropout, train)
  }
}

pub struct FineBlock {
  ln_1: MyLayerNorm,
  attn: NonCausalSelfAttention,
  ln_2: MyLayerNorm,
  mlp: MLP,
}

impl FineBlock {
  fn new(p: &nn::Path, config: &Config, layer_idx: u64) -> Self{
    let path = p / layer_idx;
    Self {
      ln_1: layer_norm(&path / "ln_1", config.n_embd, config.bias),
      ln_2: layer_norm(&path / "ln_2", config.n_embd, config.bias),
      attn: NonCausalSelfAttention::new(&(&path/ "attn"), config),
      mlp: MLP {
        c_fc: nn::linear(&path / "mlp" / "c_fc", config.n_embd, 4 * config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() }),
        c_proj: nn::linear(&path / "mlp" / "c_proj", 4 * config.n_embd, config.n_embd, LinearConfig { bias: config.bias, ..LinearConfig::default() }),
        drop: config.dropout,
      }
    }
  }

  fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
    let x = xs + self.attn.forward_t(&self.ln_1.forward_t(&xs, train), train);
    let x = &x + self.mlp.forward_t(&self.ln_2.forward_t(&x, train), train);
    return x
  }

}

pub struct FineGPT {
  pub config: Config,
  pub n_codes_total: i64,
  pub wtes: Vec<nn::Embedding>,
  pub wpe: nn::Embedding,
  pub drop: f64,
  pub h: Vec<FineBlock>,
  pub ln_f: MyLayerNorm,
  pub lm_heads: Vec<nn::Linear>,
}

impl FineGPT {
    pub fn new(p: nn::Path, config: Config) -> Self {
      let n_codes_total = config.n_codes_total;
      let transformer = &p / "transformer";
      let mut tok_embs = vec![];
      for index in 0..n_codes_total {
        let each = nn::embedding(&transformer / "wtes" / index, config.input_vocab_size, config.n_embd, Default::default());

        info!("wte index: {index} shape {:?}", each.ws.size());
        tok_embs.push(each);
      }
      let pos_emb = nn::embedding(&transformer / "wpe", config.block_size, config.n_embd, Default::default());
      let drop = config.dropout;
  
      let mut blocks = vec![];
      for block_idx in 0..config.n_layer {
          blocks.push(FineBlock::new(&(&transformer / "h"), &config, block_idx));
      }
      
      let ln_f = layer_norm(&transformer / "ln_f", config.n_embd, false);

      let mut lm_heads = vec![];
      for index in 0..n_codes_total {
        let each = nn::linear(&p /  "lm_heads" / index, config.n_embd, config.output_vocab_size, LinearConfig { bias: false, ..LinearConfig::default() });
        
        info!("lm_head index: {index} shape {:?}", each.ws.size());
        lm_heads.push(each);
      }
      
      for i in 0..(n_codes_total- config.n_codes_given) as usize  {
        tok_embs[i + 1].ws = lm_heads[i].ws.shallow_clone();
      }
      
      info!("pos_emb shape {:?}", pos_emb.ws.size());
      // info!("ln_f shape {:?}", if ln_f.ws.is_some() { ln_f.ws.unwrap().size()} else { vec![] });
      FineGPT {
        config,
        n_codes_total,
        wtes: tok_embs,
        wpe: pos_emb,
        drop,
        h: blocks,
        ln_f,
        lm_heads
      }
    }

    pub fn forward_t(&self, pred_idx: i64, idx: &Tensor, train: bool) -> Tensor {
      let (b,mut t, codes) = idx.size3().unwrap();

      assert!(t <= self.config.block_size, "Cannot forward sequence of length {t}, block size is only {}", self.config.block_size);
      assert!( pred_idx > 0, "cannot predict 0th codebook");
      assert!( codes == self.n_codes_total, "b: {b}, t: {t}, codes: {codes}");

      let pos = Tensor::arange_start(0, t, (Kind::Int64, idx.device()));

      let mut tok_embs = vec![];
      for (index, wte) in self.wtes.iter().enumerate() {
        let x = idx.i((.., .., index as i64));
        let emb = wte.forward_t(&x, train).unsqueeze(-1);
        tok_embs.push(emb);
      } // token embeddings of shape (b, t, n_embd)
      let tok_emb = Tensor::cat(&tok_embs, -1);
      let pos_emb = self.wpe.forward_t(&pos, train); // position embeddings of shape (1, t, n_embd)

      // TODO: check this sum
      let x = tok_emb.i((.., .., .., .. (pred_idx + 1))).sum(Some(Kind::Float));
      let mut x = (x + pos_emb).dropout(self.drop, train);

      for block in self.h.iter() {
        x = block.forward_t(&x, train);
      }

      x = self.ln_f.forward_t(&x, train);
      let logits = self.lm_heads[(pred_idx - self.config.n_codes_given) as usize].forward_t(&x, train);
      logits
    }
}

