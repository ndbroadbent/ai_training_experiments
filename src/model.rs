//! A small decoder-only transformer (nanoGPT-style).
//!
//! This is the **baseline / control group**: a model that learns by stuffing
//! everything — language *and* facts — into its weights. The RISC experiments
//! later are measured against it. The forward pass is written out by hand
//! (rather than using `burn::nn::attention`) so the attention and FFN blocks are
//! easy to read and to swap out for the 4D-grid experiment.

use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};

/// Hyperparameters for the baseline transformer. `#[derive(Config)]` gives us
/// `new(vocab_size)`, `with_*` builders, defaults, and JSON save/load for free.
#[derive(Config, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    #[config(default = 6)]
    pub n_layer: usize,
    #[config(default = 6)]
    pub n_head: usize,
    #[config(default = 384)]
    pub d_model: usize,
    #[config(default = 256)]
    pub block_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// FFN hidden = ff_mult * d_model. Lower it (e.g. 1-2) for the "RISC" model
    /// to shrink the feed-forward layers where factual memory tends to live.
    #[config(default = 4)]
    pub ff_mult: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        assert!(
            self.d_model % self.n_head == 0,
            "d_model ({}) must be divisible by n_head ({})",
            self.d_model,
            self.n_head
        );
        let blocks = (0..self.n_layer)
            .map(|_| Block::new(self.d_model, self.n_head, self.ff_mult, self.dropout, device))
            .collect();
        Model {
            tok_emb: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            pos_emb: EmbeddingConfig::new(self.block_size, self.d_model).init(device),
            blocks,
            ln_f: LayerNormConfig::new(self.d_model).init(device),
            head: LinearConfig::new(self.d_model, self.vocab_size)
                .with_bias(false)
                .init(device),
            block_size: self.block_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    head: Linear<B>,
    block_size: usize,
}

impl<B: Backend> Model<B> {
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// `idx`: token ids `[batch, seq]`. Returns logits `[batch, seq, vocab]`.
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, t] = idx.dims();
        let device = idx.device();

        let tok = self.tok_emb.forward(idx); // [b, t, d]
        let pos = Tensor::<B, 1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        let pos = self.pos_emb.forward(pos); // [1, t, d]
        let mut x = tok + pos; // broadcast over batch

        let mask = causal_mask::<B>(t, &device); // [1, 1, t, t]
        for block in &self.blocks {
            x = block.forward(x, mask.clone());
        }
        let x = self.ln_f.forward(x);
        self.head.forward(x) // [b, t, vocab]
    }
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    ln1: LayerNorm<B>,
    attn: SelfAttention<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    fn new(d_model: usize, n_head: usize, ff_mult: usize, dropout: f64, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(d_model).init(device),
            attn: SelfAttention::new(d_model, n_head, dropout, device),
            ln2: LayerNormConfig::new(d_model).init(device),
            mlp: Mlp::new(d_model, ff_mult, dropout, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4>) -> Tensor<B, 3> {
        // Pre-norm residual blocks (GPT-2 style).
        let x = x.clone() + self.attn.forward(self.ln1.forward(x), mask);
        x.clone() + self.mlp.forward(self.ln2.forward(x))
    }
}

#[derive(Module, Debug)]
struct SelfAttention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    dropout: Dropout,
    n_head: usize,
}

impl<B: Backend> SelfAttention<B> {
    fn new(d_model: usize, n_head: usize, dropout: f64, device: &B::Device) -> Self {
        Self {
            qkv: LinearConfig::new(d_model, 3 * d_model)
                .with_bias(false)
                .init(device),
            proj: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            dropout: DropoutConfig::new(dropout).init(),
            n_head,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        let nh = self.n_head;
        let hd = c / nh;

        let qkv = self.qkv.forward(x); // [b, t, 3c]
        let parts = qkv.chunk(3, 2); // 3 x [b, t, c]
        // [b, t, c] -> [b, nh, t, hd]
        let q = parts[0].clone().reshape([b, t, nh, hd]).swap_dims(1, 2);
        let k = parts[1].clone().reshape([b, t, nh, hd]).swap_dims(1, 2);
        let v = parts[2].clone().reshape([b, t, nh, hd]).swap_dims(1, 2);

        // Scaled dot-product attention with a causal mask.
        let scale = 1.0 / (hd as f64).sqrt();
        let att = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale); // [b, nh, t, t]
        let att = att + mask; // -inf above the diagonal
        let att = softmax(att, 3);
        let att = self.dropout.forward(att);

        let y = att.matmul(v); // [b, nh, t, hd]
        let y = y.swap_dims(1, 2).reshape([b, t, c]); // [b, t, c]
        self.proj.forward(y)
    }
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    fn new(d_model: usize, ff_mult: usize, dropout: f64, device: &B::Device) -> Self {
        let hidden = ff_mult * d_model;
        Self {
            fc1: LinearConfig::new(d_model, hidden).init(device),
            fc2: LinearConfig::new(hidden, d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = gelu(self.fc1.forward(x));
        self.dropout.forward(self.fc2.forward(x))
    }
}

/// Additive causal mask shaped `[1, 1, t, t]`: 0 on/below the diagonal,
/// -inf above it, so each position can only attend to itself and the past.
fn causal_mask<B: Backend>(t: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [t, t]), device).reshape([1, 1, t, t])
}
