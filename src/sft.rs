//! Supervised fine-tuning (SFT) on the synthetic examples.
//!
//! Differs from baseline pretraining in two ways:
//!  - Trains on rendered `Example`s (context/question/answer/tool format), not
//!    raw contiguous text.
//!  - Uses a **loss mask**: no loss on the prompt (context+question), and — key
//!    to the RISC idea — no loss on injected `>>tool:result(...)` regions, so the
//!    model is never trained to *do* arithmetic, only to call the tool.
//!
//! Typically run with `--init checkpoints/baseline` so it starts from a model
//! that already learned language, then learns the new skills.

use crate::backend::{device, Raw, B, NAME};
use crate::data::format::{self, Example, ANSWER, TOOL_RESULT_OPEN};
use crate::data::tokenizer::CharTokenizer;
use crate::data::{load_tokenizer, save_tokenizer};
use crate::model::{Model, ModelConfig};
use anyhow::{anyhow, Result};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::log_softmax;
use rand::Rng;

pub struct Params {
    pub data: String,
    pub out: String,
    /// Optional pretrained checkpoint to initialize from (its arch + tokenizer).
    pub init: Option<String>,
    pub steps: usize,
    pub batch: usize,
    pub block: usize,
    pub lr: f64,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_model: usize,
    pub ff_mult: usize,
    pub dropout: f64,
    pub eval_every: usize,
}

/// A tokenized example with a per-token "compute loss here" mask.
struct Tokenized {
    ids: Vec<i32>,
    mask: Vec<bool>,
}

pub fn run_sft(p: Params) -> Result<()> {
    let device = device();
    println!("Backend: {NAME}");

    let train_ex = format::read_jsonl(&format!("{}/train.jsonl", p.data))?;
    let val_ex = format::read_jsonl(&format!("{}/val.jsonl", p.data))?;

    // Initialize either from a pretrained checkpoint (recommended) or fresh.
    let (mut model, tok, config, block): (Model<B>, CharTokenizer, ModelConfig, usize) =
        if let Some(init) = &p.init {
            let tok = load_tokenizer(&format!("{init}/tokenizer.json"))?;
            let config = ModelConfig::load(format!("{init}/config.json"))
                .map_err(|e| anyhow!("loading init config: {e:?}"))?;
            let model: Model<B> = config
                .init(&device)
                .load_file(format!("{init}/model"), &CompactRecorder::new(), &device)
                .map_err(|e| anyhow!("loading init weights: {e:?}"))?;
            let block = config.block_size; // architecture is fixed by the checkpoint
            println!(
                "Initialized from {init}: {} params, block {block}, vocab {}",
                model.num_params(),
                tok.vocab_size()
            );
            (model, tok, config, block)
        } else {
            let tok = build_tokenizer(&train_ex);
            let config = ModelConfig::new(tok.vocab_size())
                .with_n_layer(p.n_layer)
                .with_n_head(p.n_head)
                .with_d_model(p.d_model)
                .with_block_size(p.block)
                .with_ff_mult(p.ff_mult)
                .with_dropout(p.dropout);
            let model = config.init(&device);
            println!(
                "Fresh model: {} params, block {}, vocab {}",
                model.num_params(),
                p.block,
                tok.vocab_size()
            );
            (model, tok, config, p.block)
        };

    // Keep only examples that fit the context window intact — otherwise the
    // answer (at the end) would be truncated away and the example is useless.
    let mut train = tokenize_all(&train_ex, &tok);
    let mut val = tokenize_all(&val_ex, &tok);
    let train_before = train.len();
    train.retain(|t| t.ids.len() <= block);
    val.retain(|t| t.ids.len() <= block);
    println!(
        "Usable train examples: {} (dropped {} over block {block})  val: {}",
        train.len(),
        train_before - train.len(),
        val.len()
    );

    let mut optim = AdamWConfig::new().init();
    std::fs::create_dir_all(&p.out)?;
    config
        .save(format!("{}/config.json", p.out))
        .map_err(|e| anyhow!("saving config: {e:?}"))?;
    save_tokenizer(&tok, &format!("{}/tokenizer.json", p.out))?;

    let mut rng = rand::thread_rng();
    for step in 1..=p.steps {
        let (x, y, m) = make_batch::<B>(&train, p.batch, block, &device, &mut rng);
        let loss = masked_loss::<B>(model.forward(x), y, m);

        let report = step == 1 || step % p.eval_every == 0;
        let tl = report.then(|| loss.clone().into_scalar().elem::<f32>());

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(p.lr, model, grads);

        if let Some(tl) = tl {
            let vl = eval_masked(&model, &val, p.batch, block, 20, &device);
            println!("step {step:>5}  train {tl:.4}  val {vl:.4}");
        }
    }

    model
        .clone()
        .save_file(format!("{}/model", p.out), &CompactRecorder::new())
        .map_err(|e| anyhow!("saving model: {e:?}"))?;
    println!("\nSaved SFT checkpoint to {}/", p.out);
    Ok(())
}

fn build_tokenizer(examples: &[Example]) -> CharTokenizer {
    let mut all = String::new();
    for e in examples {
        all.push_str(&e.render());
    }
    CharTokenizer::build(&all)
}

fn tokenize_all(examples: &[Example], tok: &CharTokenizer) -> Vec<Tokenized> {
    examples
        .iter()
        .map(|e| tokenize(e, tok))
        .filter(|t| t.ids.len() >= 2)
        .collect()
}

fn tokenize(ex: &Example, tok: &CharTokenizer) -> Tokenized {
    let rendered = ex.render();
    let char_mask = compute_mask(&rendered);
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for (ch, keep) in rendered.chars().zip(char_mask) {
        if let Some(id) = tok.id(ch) {
            ids.push(id as i32);
            mask.push(keep);
        }
    }
    Tokenized { ids, mask }
}

/// Per-character loss mask over a rendered example: true where the model should
/// be trained to predict the character. Prompt = false; answer = true; injected
/// `>>tool:result(...)` regions (and their surrounding newlines) = false.
fn compute_mask(rendered: &str) -> Vec<bool> {
    let total = rendered.chars().count();
    let mut mask = vec![false; total];

    // Everything after the answer marker is, by default, learned.
    let ans = format!("{ANSWER}\n");
    if let Some(b) = rendered.find(&ans) {
        let start_char = rendered[..b + ans.len()].chars().count();
        for m in mask.iter_mut().skip(start_char) {
            *m = true;
        }
    }

    // Turn loss off over each injected tool-result span.
    let mut from = 0;
    while let Some(rel) = rendered[from..].find(TOOL_RESULT_OPEN) {
        let open_byte = from + rel;
        let Some(crel) = rendered[open_byte..].find(')') else {
            break;
        };
        let close_byte = open_byte + crel;
        let mut s = rendered[..open_byte].chars().count();
        let mut e = rendered[..=close_byte].chars().count(); // exclusive, includes ')'
        if s > 0 && rendered.chars().nth(s - 1) == Some('\n') {
            s -= 1; // also drop the newline before the result
        }
        if e < total && rendered.chars().nth(e) == Some('\n') {
            e += 1; // and the newline after it
        }
        for m in mask.iter_mut().take(e.min(total)).skip(s) {
            *m = false;
        }
        from = close_byte + 1;
    }
    mask
}

/// Sample a batch, padding to the longest example in the batch (capped at
/// `block`). Inputs/targets are next-token shifted; pad positions get mask 0.
fn make_batch<BK: Backend>(
    data: &[Tokenized],
    batch: usize,
    block: usize,
    device: &BK::Device,
    rng: &mut impl Rng,
) -> (Tensor<BK, 2, Int>, Tensor<BK, 2, Int>, Tensor<BK, 2>) {
    let chosen: Vec<&Tokenized> = (0..batch).map(|_| &data[rng.gen_range(0..data.len())]).collect();
    let max_len = chosen
        .iter()
        .map(|s| s.ids.len().min(block))
        .max()
        .unwrap_or(2)
        .max(2);
    let t = max_len - 1; // length after the next-token shift

    let mut xs = vec![0i32; batch * t];
    let mut ys = vec![0i32; batch * t];
    let mut ms = vec![0f32; batch * t];
    for (bi, s) in chosen.iter().enumerate() {
        let len = s.ids.len().min(block);
        for j in 0..len.saturating_sub(1) {
            xs[bi * t + j] = s.ids[j];
            ys[bi * t + j] = s.ids[j + 1];
            ms[bi * t + j] = if s.mask[j + 1] { 1.0 } else { 0.0 };
        }
    }
    (
        Tensor::from_data(TensorData::new(xs, [batch, t]), device),
        Tensor::from_data(TensorData::new(ys, [batch, t]), device),
        Tensor::from_data(TensorData::new(ms, [batch, t]), device),
    )
}

/// Cross-entropy averaged over only the masked (answer) positions.
fn masked_loss<BK: Backend>(
    logits: Tensor<BK, 3>,
    targets: Tensor<BK, 2, Int>,
    mask: Tensor<BK, 2>,
) -> Tensor<BK, 1> {
    let [b, t, v] = logits.dims();
    let n = b * t;
    let logp = log_softmax(logits.reshape([n, v]), 1);
    let picked = logp.gather(1, targets.reshape([n, 1])).reshape([n]);
    let mask = mask.reshape([n]);
    let nll = picked.neg().mul(mask.clone());
    nll.sum().div(mask.sum().clamp_min(1.0))
}

fn eval_masked(
    model: &Model<B>,
    data: &[Tokenized],
    batch: usize,
    block: usize,
    iters: usize,
    device: &<Raw as Backend>::Device,
) -> f32 {
    if data.is_empty() {
        return f32::NAN;
    }
    let model = model.valid();
    let mut rng = rand::thread_rng();
    let mut total = 0.0;
    for _ in 0..iters {
        let (x, y, m) = make_batch::<Raw>(data, batch, block, device, &mut rng);
        total += masked_loss::<Raw>(model.forward(x), y, m).into_scalar().elem::<f32>();
    }
    total / iters as f32
}
