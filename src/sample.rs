//! Autoregressive text generation + the `sample` subcommand.

use crate::backend::{device, Raw};
use crate::data::{self, format, tokenizer::CharTokenizer};
use crate::model::{Model, ModelConfig};
use crate::retrieval::Bm25Index;
use crate::tools::calculator;
use anyhow::{anyhow, Result};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::softmax;
use rand::Rng;
use std::path::Path;

pub struct Params {
    pub dir: String,
    pub prompt: String,
    pub max_new: usize,
    pub temperature: f64,
}

/// Generate `max_new` tokens from `model`, continuing `prompt`. Inference runs on
/// the raw (non-autodiff) backend, so dropout is off and no graph is built.
pub fn generate(
    model: &Model<Raw>,
    tok: &CharTokenizer,
    prompt: &str,
    max_new: usize,
    temperature: f64,
    device: &<Raw as Backend>::Device,
    rng: &mut impl Rng,
) -> String {
    let block_size = model.block_size();
    let mut ids: Vec<i32> = tok.encode(prompt).into_iter().map(|x| x as i32).collect();
    if ids.is_empty() {
        ids.push(0); // seed with the first vocab symbol when no prompt is given
    }
    let start_len = ids.len();
    let vocab = tok.vocab_size();

    for _ in 0..max_new {
        let ctx_len = ids.len().min(block_size);
        let ctx = ids[ids.len() - ctx_len..].to_vec();
        let x = Tensor::<Raw, 2, Int>::from_data(TensorData::new(ctx, [1, ctx_len]), device);

        let logits = model.forward(x); // [1, ctx_len, vocab]
        let last = logits
            .slice([0..1, ctx_len - 1..ctx_len, 0..vocab])
            .reshape([vocab])
            .mul_scalar(1.0 / temperature.max(1e-6));
        let probs: Vec<f32> = softmax(last, 0).into_data().to_vec().unwrap();

        // Multinomial sampling via inverse-CDF.
        let r: f32 = rng.gen();
        let mut cum = 0.0;
        let mut next = vocab - 1;
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                next = i;
                break;
            }
        }
        ids.push(next as i32);
    }

    let out: Vec<u32> = ids[start_len..].iter().map(|&x| x as u32).collect();
    tok.decode(&out)
}

pub fn run_sample(params: Params) -> Result<()> {
    let device = device();
    let config = ModelConfig::load(format!("{}/config.json", params.dir))
        .map_err(|e| anyhow!("loading config: {e:?}"))?;
    let tok = data::load_tokenizer(&format!("{}/tokenizer.json", params.dir))?;

    let model: Model<Raw> = config
        .init(&device)
        .load_file(
            format!("{}/model", params.dir),
            &CompactRecorder::new(),
            &device,
        )
        .map_err(|e| anyhow!("loading model weights: {e:?}"))?;

    let mut rng = rand::thread_rng();
    let text = generate(
        &model,
        &tok,
        &params.prompt,
        params.max_new,
        params.temperature,
        &device,
        &mut rng,
    );

    if !params.prompt.is_empty() {
        print!("{}", params.prompt);
    }
    println!("{text}");
    Ok(())
}

/// Inverse-CDF multinomial sampling from a probability vector.
fn sample_multinomial(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i;
        }
    }
    probs.len() - 1
}

/// If `s` ends with a complete, paren-balanced `>>tool:calculate(<expr>)`,
/// return `<expr>`. Used to intercept a tool call the instant it's emitted.
fn pending_tool_call(s: &str) -> Option<String> {
    let marker = ">>tool:calculate(";
    let open = s.rfind(marker)? + marker.len();
    let mut depth = 1i32;
    let mut expr = String::new();
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => {
                depth += 1;
                expr.push(c);
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    // Trigger only if this ')' is the very last character emitted.
                    return (open + i == s.len() - 1).then_some(expr);
                }
                expr.push(c);
            }
            _ => expr.push(c),
        }
    }
    None
}

/// Generate an answer for an already-formatted prompt (text up to and including
/// `<|answer|>\n`), running the calculator tool-loop: when the model emits a
/// `>>tool:calculate(...)` call, evaluate it, inject `>>tool:result(...)`, and
/// continue. Stops at the `<|end|>` marker or after `max_new` model tokens.
pub fn generate_instruct(
    model: &Model<Raw>,
    tok: &CharTokenizer,
    prompt: &str,
    max_new: usize,
    temperature: f64,
    device: &<Raw as Backend>::Device,
    rng: &mut impl Rng,
) -> String {
    let block = model.block_size();
    let vocab = tok.vocab_size();
    let mut ids: Vec<i32> = tok.encode(prompt).into_iter().map(|x| x as i32).collect();
    let mut produced = String::new();

    let encode = |s: &str| -> Vec<i32> { tok.encode(s).into_iter().map(|x| x as i32).collect() };

    for _ in 0..max_new {
        let ctx_len = ids.len().min(block);
        let ctx = ids[ids.len() - ctx_len..].to_vec();
        let x = Tensor::<Raw, 2, Int>::from_data(TensorData::new(ctx, [1, ctx_len]), device);

        let last = model
            .forward(x)
            .slice([0..1, ctx_len - 1..ctx_len, 0..vocab])
            .reshape([vocab])
            .mul_scalar(1.0 / temperature.max(1e-6));
        let probs: Vec<f32> = softmax(last, 0).into_data().to_vec().unwrap();
        let next = sample_multinomial(&probs, rng);

        ids.push(next as i32);
        produced.push_str(&tok.decode(&[next as u32]));

        if let Some(cut) = produced.find(format::END) {
            produced.truncate(cut);
            break;
        }

        // Intercept and resolve a completed tool call.
        if let Some(expr) = pending_tool_call(&produced) {
            let injected = match calculator::eval(&expr) {
                Ok(v) => format!("\n>>tool:result({})\n", calculator::format_number(v)),
                Err(_) => "\n>>tool:result(error)\n".to_string(),
            };
            ids.extend(encode(&injected));
            produced.push_str(&injected);
        }
    }
    produced
}

pub struct AskParams {
    pub dir: String,
    pub query: String,
    pub notes: String,
    pub retrieve: bool,
    pub top_k: usize,
    pub chunk_words: usize,
    pub max_new: usize,
    pub temperature: f64,
}

/// `ask` subcommand: the capstone. Optionally retrieve context from the vault,
/// format the prompt, and answer with the tool-loop — RAG + tools end to end.
pub fn run_ask(p: AskParams) -> Result<()> {
    let device = device();
    let config = ModelConfig::load(format!("{}/config.json", p.dir))
        .map_err(|e| anyhow!("loading config: {e:?}"))?;
    let tok = data::load_tokenizer(&format!("{}/tokenizer.json", p.dir))?;
    let model: Model<Raw> = config
        .init(&device)
        .load_file(format!("{}/model", p.dir), &CompactRecorder::new(), &device)
        .map_err(|e| anyhow!("loading model: {e:?}"))?;

    let context = if p.retrieve {
        let index = Bm25Index::from_vault(Path::new(&p.notes), p.chunk_words)?;
        let hits = index.search(&p.query, p.top_k);
        hits.iter()
            .map(|h| h.snippet.clone())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        String::new()
    };

    let mut prompt = String::new();
    if !context.trim().is_empty() {
        prompt.push_str(format::CTX);
        prompt.push('\n');
        prompt.push_str(context.trim());
        prompt.push('\n');
    }
    prompt.push_str(format::QUESTION);
    prompt.push('\n');
    prompt.push_str(p.query.trim());
    prompt.push('\n');
    prompt.push_str(format::ANSWER);
    prompt.push('\n');

    let mut rng = rand::thread_rng();
    let answer = generate_instruct(&model, &tok, &prompt, p.max_new, p.temperature, &device, &mut rng);

    if p.retrieve {
        println!("[retrieved {} chars of context]\n", context.len());
    }
    println!("Q: {}\nA: {}", p.query.trim(), answer.trim());
    Ok(())
}
