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
use burn::module::Ignored;
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

    // --- Experimental N-D neuron-grid block (replaces the FFN when enabled) ---
    /// Use the grid block instead of the MLP feed-forward layer.
    #[config(default = false)]
    pub use_grid: bool,
    /// Lattice dimensionality (3 = cube, 5 = your 5D idea).
    #[config(default = 5)]
    pub grid_dims: usize,
    /// Cells per axis. Total cells = grid_side ^ grid_dims. Use >=4 for true
    /// locality; side 3 makes every cell a neighbor of every other.
    #[config(default = 3)]
    pub grid_side: usize,
    /// Feature channels per cell. Pick so cells*channels ~= ff_mult*d_model.
    #[config(default = 6)]
    pub grid_channels: usize,
    /// Number of local-update iterations (the "depth" of the grid).
    #[config(default = 3)]
    pub grid_iters: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        assert!(
            self.d_model % self.n_head == 0,
            "d_model ({}) must be divisible by n_head ({})",
            self.d_model,
            self.n_head
        );
        let blocks = (0..self.n_layer).map(|_| Block::new(self, device)).collect();
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

    /// Like [`forward`], but also returns the residual-stream activation vector
    /// for the *last* position after each layer: `[embedding, block_1, ...,
    /// block_n, final_norm]`. Used by the visualizer to watch the net "fire".
    pub fn forward_capture(&self, idx: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Vec<Vec<f32>>) {
        let [_, t] = idx.dims();
        let device = idx.device();

        let tok = self.tok_emb.forward(idx);
        let pos = Tensor::<B, 1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        let mut x = tok + self.pos_emb.forward(pos);

        let mut acts = vec![last_row(&x)];
        let mask = causal_mask::<B>(t, &device);
        for block in &self.blocks {
            x = block.forward(x, mask.clone());
            acts.push(last_row(&x));
        }
        let x = self.ln_f.forward(x);
        acts.push(last_row(&x));
        (self.head.forward(x), acts)
    }
}

/// Extract the last-position feature vector of a `[batch, seq, d]` tensor as a
/// host `Vec<f32>`.
fn last_row<B: Backend>(x: &Tensor<B, 3>) -> Vec<f32> {
    let [_, t, d] = x.dims();
    x.clone()
        .slice([0..1, t - 1..t, 0..d])
        .reshape([d])
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    ln1: LayerNorm<B>,
    attn: SelfAttention<B>,
    ln2: LayerNorm<B>,
    // Exactly one of these is populated, chosen by `config.use_grid`.
    mlp: Option<Mlp<B>>,
    grid: Option<GridBlock<B>>,
}

impl<B: Backend> Block<B> {
    fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let (mlp, grid) = if config.use_grid {
            (None, Some(GridBlock::new(config, device)))
        } else {
            let mlp = Mlp::new(config.d_model, config.ff_mult, config.dropout, device);
            (Some(mlp), None)
        };
        Self {
            ln1: LayerNormConfig::new(config.d_model).init(device),
            attn: SelfAttention::new(config.d_model, config.n_head, config.dropout, device),
            ln2: LayerNormConfig::new(config.d_model).init(device),
            mlp,
            grid,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4>) -> Tensor<B, 3> {
        // Pre-norm residual blocks (GPT-2 style). The feed-forward sublayer is
        // either the standard MLP or the experimental grid block.
        let x = x.clone() + self.attn.forward(self.ln1.forward(x), mask);
        let normed = self.ln2.forward(x.clone());
        let ff = match &self.mlp {
            Some(mlp) => mlp.forward(normed),
            None => self.grid.as_ref().expect("mlp or grid must be set").forward(normed),
        };
        x + ff
    }
}

/// Experimental feed-forward replacement: an N-dimensional lattice of cells with
/// Moore connectivity (orthogonal + diagonal + cross-dimension neighbors). A
/// token's vector is projected into the grid, the cells update from their
/// neighbors for `iters` iterations (toroidal boundaries; diagonals give
/// Chebyshev-distance mixing, so information spreads fast), then it's projected
/// back. The lattice topology lives entirely in the (constant) adjacency matrix.
#[derive(Module, Debug)]
struct GridBlock<B: Backend> {
    in_proj: Linear<B>,
    self_lin: Linear<B>,
    nbr_lin: Linear<B>,
    out_proj: Linear<B>,
    /// Row-normalized [cells, cells] neighbor matrix, flattened. Backend-agnostic
    /// constant data; materialized into a tensor per forward (cheap for small
    /// grids). Not a trained parameter.
    adjacency: Ignored<Vec<f32>>,
    num_cells: usize,
    channels: usize,
    iters: usize,
}

impl<B: Backend> GridBlock<B> {
    fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let dims = config.grid_dims;
        let side = config.grid_side;
        let channels = config.grid_channels;
        let num_cells = side.pow(dims as u32);
        let state = num_cells * channels;
        Self {
            in_proj: LinearConfig::new(config.d_model, state).init(device),
            self_lin: LinearConfig::new(channels, channels).init(device),
            nbr_lin: LinearConfig::new(channels, channels).init(device),
            out_proj: LinearConfig::new(state, config.d_model).init(device),
            adjacency: Ignored(moore_adjacency_data(dims, side)),
            num_cells,
            channels,
            iters: config.grid_iters,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        let m = b * t;
        // Project each token vector into a grid state [tokens, cells, channels].
        let mut s = self
            .in_proj
            .forward(x.reshape([m, d]))
            .reshape([m, self.num_cells, self.channels]);

        for _ in 0..self.iters {
            let agg = self.aggregate(s.clone());
            // Each cell mixes its own state with the mean of its neighbors.
            let update = gelu(self.self_lin.forward(s.clone()) + self.nbr_lin.forward(agg));
            s = s + update;
        }

        self.out_proj
            .forward(s.reshape([m, self.num_cells * self.channels]))
            .reshape([b, t, d])
    }

    /// Mean of each cell's neighbors, as `adjacency @ state` over the cell axis.
    fn aggregate(&self, s: Tensor<B, 3>) -> Tensor<B, 3> {
        let [m, nc, c] = s.dims();
        let device = s.device();
        let a = Tensor::<B, 2>::from_data(TensorData::new(self.adjacency.0.clone(), [nc, nc]), &device);
        let s_perm = s.swap_dims(0, 1).reshape([nc, m * c]); // [cells, tokens*channels]
        let agg = a.matmul(s_perm); // [cells, tokens*channels]
        agg.reshape([nc, m, c]).swap_dims(0, 1) // [tokens, cells, channels]
    }
}

/// Build a row-normalized Moore-neighborhood adjacency matrix for a `dims`-D
/// lattice with `side` cells per axis and toroidal (wrap-around) boundaries.
/// Each cell connects to all cells differing by -1/0/+1 per axis (excluding
/// itself): orthogonal, diagonal, and across dimensions.
fn moore_adjacency_data(dims: usize, side: usize) -> Vec<f32> {
    let nc = side.pow(dims as u32);
    let offsets = moore_offsets(dims);
    let mut a = vec![0f32; nc * nc];

    for i in 0..nc {
        let coords = decode(i, dims, side);
        for off in &offsets {
            let neighbor: Vec<usize> = (0..dims)
                .map(|d| (coords[d] as i64 + off[d]).rem_euclid(side as i64) as usize)
                .collect();
            let j = encode(&neighbor, side);
            if j != i {
                a[i * nc + j] += 1.0;
            }
        }
        let sum: f32 = a[i * nc..(i + 1) * nc].iter().sum();
        if sum > 0.0 {
            for v in &mut a[i * nc..(i + 1) * nc] {
                *v /= sum;
            }
        }
    }
    a
}

/// All offset vectors in {-1,0,1}^dims except the all-zero (self) vector.
fn moore_offsets(dims: usize) -> Vec<Vec<i64>> {
    let mut result = vec![vec![]];
    for _ in 0..dims {
        result = result
            .into_iter()
            .flat_map(|prefix| {
                [-1i64, 0, 1].into_iter().map(move |v| {
                    let mut p = prefix.clone();
                    p.push(v);
                    p
                })
            })
            .collect();
    }
    result.into_iter().filter(|o| o.iter().any(|&x| x != 0)).collect()
}

fn decode(mut i: usize, dims: usize, side: usize) -> Vec<usize> {
    let mut coords = vec![0; dims];
    for c in coords.iter_mut() {
        *c = i % side;
        i /= side;
    }
    coords
}

fn encode(coords: &[usize], side: usize) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for &c in coords {
        idx += c * stride;
        stride *= side;
    }
    idx
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
