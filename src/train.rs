//! Training loop for the baseline transformer.
//!
//! Deliberately a hand-written loop (not `burn-train`'s `Learner`) so every step
//! — batching, loss, backward, optimizer step, eval — is visible and easy to
//! mutate when we start experimenting with architectures.

use crate::backend::{device, Raw, B, NAME};
use crate::data;
use crate::model::{Model, ModelConfig};
use crate::sample;
use anyhow::{anyhow, Result};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use rand::Rng;
use std::path::Path;

pub struct Params {
    pub notes: String,
    pub out: String,
    pub steps: usize,
    pub batch: usize,
    pub block: usize,
    pub lr: f64,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_model: usize,
    pub dropout: f64,
    pub eval_every: usize,
    pub eval_iters: usize,
}

pub fn run_train(p: Params) -> Result<()> {
    let device = device();
    println!("Backend: {NAME}");

    // --- data ---------------------------------------------------------------
    let (tokens, tok) = data::load_tokens(Path::new(&p.notes))?;
    let n = tokens.len();
    let split = (n as f64 * 0.9) as usize;
    let (train_data, val_data) = tokens.split_at(split);
    println!(
        "Tokens: {n} (train {} / val {})  Vocab: {}",
        train_data.len(),
        val_data.len(),
        tok.vocab_size()
    );

    // --- model --------------------------------------------------------------
    let config = ModelConfig::new(tok.vocab_size())
        .with_n_layer(p.n_layer)
        .with_n_head(p.n_head)
        .with_d_model(p.d_model)
        .with_block_size(p.block)
        .with_dropout(p.dropout);
    let mut model: Model<B> = config.init(&device);
    println!(
        "Model: {} layers, {} heads, d_model {}, block {}  ->  {} params",
        p.n_layer,
        p.n_head,
        p.d_model,
        p.block,
        model.num_params()
    );

    let mut optim = AdamWConfig::new().init();

    // --- persist config + tokenizer so `sample` can reload ------------------
    std::fs::create_dir_all(&p.out)?;
    config
        .save(format!("{}/config.json", p.out))
        .map_err(|e| anyhow!("saving config: {e:?}"))?;
    data::save_tokenizer(&tok, &format!("{}/tokenizer.json", p.out))?;

    // --- train --------------------------------------------------------------
    let mut rng = rand::thread_rng();
    for step in 1..=p.steps {
        let (x, y) = get_batch::<B>(train_data, p.block, p.batch, &device, &mut rng);
        let logits = model.forward(x);
        let loss = compute_loss::<B>(logits, y, &device);

        let report = step == 1 || step % p.eval_every == 0;
        let train_loss = if report {
            Some(loss.clone().into_scalar().elem::<f32>())
        } else {
            None
        };

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(p.lr, model, grads);

        if let Some(tl) = train_loss {
            let val_loss = estimate_loss(&model, val_data, p.block, p.batch, p.eval_iters, &device);
            println!("step {step:>6}  train {tl:.4}  val {val_loss:.4}");
        }
    }

    // --- save + a quick sample ---------------------------------------------
    model
        .clone()
        .save_file(format!("{}/model", p.out), &CompactRecorder::new())
        .map_err(|e| anyhow!("saving model: {e:?}"))?;
    println!("\nSaved checkpoint to {}/", p.out);

    let preview = sample::generate(&model.valid(), &tok, "", 400, 0.8, &device, &mut rng);
    println!("\n--- sample (400 chars) ---\n{preview}\n--- end sample ---");
    Ok(())
}

/// Sample a random batch of contiguous `block`-length token windows.
/// Returns inputs and next-token targets, both `[batch, block]`.
fn get_batch<BK: Backend>(
    data: &[i32],
    block: usize,
    batch: usize,
    device: &BK::Device,
    rng: &mut impl Rng,
) -> (Tensor<BK, 2, Int>, Tensor<BK, 2, Int>) {
    let max_start = data.len() - block - 1;
    let mut xs = Vec::with_capacity(batch * block);
    let mut ys = Vec::with_capacity(batch * block);
    for _ in 0..batch {
        let s = rng.gen_range(0..max_start);
        xs.extend_from_slice(&data[s..s + block]);
        ys.extend_from_slice(&data[s + 1..s + block + 1]);
    }
    let x = Tensor::<BK, 2, Int>::from_data(TensorData::new(xs, [batch, block]), device);
    let y = Tensor::<BK, 2, Int>::from_data(TensorData::new(ys, [batch, block]), device);
    (x, y)
}

/// Cross-entropy next-token loss over a `[batch, seq, vocab]` logits tensor.
fn compute_loss<BK: Backend>(
    logits: Tensor<BK, 3>,
    targets: Tensor<BK, 2, Int>,
    device: &BK::Device,
) -> Tensor<BK, 1> {
    let [b, t, c] = logits.dims();
    let logits = logits.reshape([b * t, c]);
    let targets = targets.reshape([b * t]);
    CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits, targets)
}

/// Average validation loss over `iters` random batches, on the non-autodiff
/// backend (no graph, no dropout).
fn estimate_loss(
    model: &Model<B>,
    data: &[i32],
    block: usize,
    batch: usize,
    iters: usize,
    device: &<Raw as Backend>::Device,
) -> f32 {
    let model = model.valid(); // Model<Raw>
    let mut rng = rand::thread_rng();
    let mut total = 0.0;
    for _ in 0..iters {
        let (x, y) = get_batch::<Raw>(data, block, batch, device, &mut rng);
        let loss = compute_loss::<Raw>(model.forward(x), y, device);
        total += loss.into_scalar().elem::<f32>();
    }
    total / iters as f32
}
