//! Toy RISC-style language-model experiments in Rust + Burn.
//!
//! The guiding philosophy: build a small model that learns *skills* (reasoning,
//! tool calls, reading retrieved context) rather than memorizing *facts*. Facts
//! live outside the weights, fetched via retrieval. See README.md for the roadmap.

mod backend;
mod checks;
mod data;
mod model;
mod retrieval;
mod sample;
mod sft;
mod tools;
mod train;
mod viz;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ai_training", about = "Toy RISC-style LM experiments (Rust + Burn)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Backend + autodiff sanity check (prints a gradient).
    Hello,
    /// Load the notes vault, clean it, build a char tokenizer, and print stats.
    Data {
        /// Path to the Obsidian vault / notes directory.
        #[arg(long)]
        notes: Option<String>,
    },
    /// Train the baseline transformer.
    Train {
        #[arg(long)]
        notes: Option<String>,
        /// Output directory for the checkpoint, config, and tokenizer.
        #[arg(long, default_value = "checkpoints/baseline")]
        out: String,
        #[arg(long, default_value_t = 2000)]
        steps: usize,
        #[arg(long, default_value_t = 32)]
        batch: usize,
        #[arg(long, default_value_t = 256)]
        block: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        #[arg(long, default_value_t = 6)]
        layers: usize,
        #[arg(long, default_value_t = 6)]
        heads: usize,
        #[arg(long = "dmodel", default_value_t = 384)]
        d_model: usize,
        #[arg(long, default_value_t = 0.1)]
        dropout: f64,
        #[arg(long = "eval-every", default_value_t = 200)]
        eval_every: usize,
        #[arg(long = "eval-iters", default_value_t = 50)]
        eval_iters: usize,
        /// Use the experimental N-D neuron-grid block instead of the MLP.
        #[arg(long = "use-grid")]
        use_grid: bool,
        #[arg(long = "grid-dims", default_value_t = 5)]
        grid_dims: usize,
        #[arg(long = "grid-side", default_value_t = 3)]
        grid_side: usize,
        #[arg(long = "grid-channels", default_value_t = 6)]
        grid_channels: usize,
        #[arg(long = "grid-iters", default_value_t = 3)]
        grid_iters: usize,
    },
    /// Generate the synthetic training dataset (calculator + cloze + seed).
    GenData {
        #[arg(long)]
        notes: Option<String>,
        /// Output directory for train.jsonl / val.jsonl.
        #[arg(long, default_value = "data")]
        out: String,
        /// Number of programmatic calculator examples.
        #[arg(long, default_value_t = 3000)]
        calc: usize,
        /// Number of programmatic cloze-QA examples from the vault.
        #[arg(long, default_value_t = 3000)]
        cloze: usize,
        #[arg(long = "chunk-words", default_value_t = 120)]
        chunk_words: usize,
        /// Hand-authored seed examples (JSONL).
        #[arg(long, default_value = "data/seed_examples.jsonl")]
        seed: String,
        /// How many times to repeat (upweight) the seed examples.
        #[arg(long = "seed-repeat", default_value_t = 30)]
        seed_repeat: usize,
    },
    /// Build a BM25 index over the vault and run one retrieval query.
    Retrieve {
        #[arg(long)]
        notes: Option<String>,
        /// The search query.
        #[arg(long)]
        query: String,
        #[arg(long = "top-k", default_value_t = 5)]
        top_k: usize,
        /// Target passage size in words.
        #[arg(long = "chunk-words", default_value_t = 120)]
        chunk_words: usize,
    },
    /// Supervised fine-tune on the synthetic dataset (RISC skills + tools).
    TrainSft {
        /// Directory containing train.jsonl / val.jsonl.
        #[arg(long, default_value = "data")]
        data: String,
        #[arg(long, default_value = "checkpoints/risc")]
        out: String,
        /// Pretrained checkpoint to initialize from (recommended).
        #[arg(long, default_value = "checkpoints/baseline")]
        init: String,
        /// Train from scratch instead of initializing from `--init`.
        #[arg(long = "from-scratch")]
        from_scratch: bool,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 16)]
        batch: usize,
        #[arg(long, default_value_t = 256)]
        block: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        #[arg(long, default_value_t = 6)]
        layers: usize,
        #[arg(long, default_value_t = 6)]
        heads: usize,
        #[arg(long = "dmodel", default_value_t = 384)]
        d_model: usize,
        #[arg(long = "ff-mult", default_value_t = 4)]
        ff_mult: usize,
        #[arg(long, default_value_t = 0.1)]
        dropout: f64,
        #[arg(long = "eval-every", default_value_t = 100)]
        eval_every: usize,
    },
    /// Ask a question: retrieve context, then answer with the calculator loop.
    Ask {
        #[arg(long, default_value = "checkpoints/risc")]
        dir: String,
        #[arg(long)]
        query: String,
        #[arg(long)]
        notes: Option<String>,
        /// Skip retrieval and answer from the question alone (e.g. pure math).
        #[arg(long = "no-retrieve")]
        no_retrieve: bool,
        #[arg(long = "top-k", default_value_t = 3)]
        top_k: usize,
        #[arg(long = "chunk-words", default_value_t = 60)]
        chunk_words: usize,
        #[arg(long = "max-new", default_value_t = 300)]
        max_new: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
    },
    /// Render an mp4 of the network firing as it answers a query.
    Viz {
        #[arg(long, default_value = "checkpoints/risc")]
        dir: String,
        #[arg(long)]
        query: String,
        #[arg(long)]
        notes: Option<String>,
        #[arg(long = "no-retrieve")]
        no_retrieve: bool,
        #[arg(long = "top-k", default_value_t = 3)]
        top_k: usize,
        #[arg(long = "chunk-words", default_value_t = 60)]
        chunk_words: usize,
        #[arg(long = "max-new", default_value_t = 160)]
        max_new: usize,
        #[arg(long, default_value = "viz.mp4")]
        out: String,
        #[arg(long, default_value_t = 10)]
        fps: usize,
    },
    /// Generate text from a trained checkpoint.
    Sample {
        /// Checkpoint directory (as passed to `train --out`).
        #[arg(long, default_value = "checkpoints/baseline")]
        dir: String,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long = "max-new", default_value_t = 500)]
        max_new: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
    },
}

fn default_notes() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    format!("{home}/Notes")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Hello => checks::run_hello(),
        Command::Data { notes } => {
            data::run_data(PathBuf::from(notes.unwrap_or_else(default_notes)))?;
        }
        Command::Train {
            notes,
            out,
            steps,
            batch,
            block,
            lr,
            layers,
            heads,
            d_model,
            dropout,
            eval_every,
            eval_iters,
            use_grid,
            grid_dims,
            grid_side,
            grid_channels,
            grid_iters,
        } => {
            train::run_train(train::Params {
                notes: notes.unwrap_or_else(default_notes),
                out,
                steps,
                batch,
                block,
                lr,
                n_layer: layers,
                n_head: heads,
                d_model,
                dropout,
                eval_every,
                eval_iters,
                use_grid,
                grid_dims,
                grid_side,
                grid_channels,
                grid_iters,
            })?;
        }
        Command::GenData {
            notes,
            out,
            calc,
            cloze,
            chunk_words,
            seed,
            seed_repeat,
        } => {
            data::synth::run_gen_data(data::synth::Params {
                notes: notes.unwrap_or_else(default_notes),
                out,
                calc,
                cloze,
                chunk_words,
                seed,
                seed_repeat,
            })?;
        }
        Command::Retrieve {
            notes,
            query,
            top_k,
            chunk_words,
        } => {
            retrieval::run_retrieve(retrieval::Params {
                notes: notes.unwrap_or_else(default_notes),
                query,
                top_k,
                chunk_words,
            })?;
        }
        Command::TrainSft {
            data,
            out,
            init,
            from_scratch,
            steps,
            batch,
            block,
            lr,
            layers,
            heads,
            d_model,
            ff_mult,
            dropout,
            eval_every,
        } => {
            sft::run_sft(sft::Params {
                data,
                out,
                init: (!from_scratch).then_some(init),
                steps,
                batch,
                block,
                lr,
                n_layer: layers,
                n_head: heads,
                d_model,
                ff_mult,
                dropout,
                eval_every,
            })?;
        }
        Command::Ask {
            dir,
            query,
            notes,
            no_retrieve,
            top_k,
            chunk_words,
            max_new,
            temperature,
        } => {
            sample::run_ask(sample::AskParams {
                dir,
                query,
                notes: notes.unwrap_or_else(default_notes),
                retrieve: !no_retrieve,
                top_k,
                chunk_words,
                max_new,
                temperature,
            })?;
        }
        Command::Viz {
            dir,
            query,
            notes,
            no_retrieve,
            top_k,
            chunk_words,
            max_new,
            out,
            fps,
        } => {
            viz::run_viz(viz::Params {
                dir,
                query,
                notes: notes.unwrap_or_else(default_notes),
                retrieve: !no_retrieve,
                top_k,
                chunk_words,
                max_new,
                out,
                fps,
            })?;
        }
        Command::Sample {
            dir,
            prompt,
            max_new,
            temperature,
        } => {
            sample::run_sample(sample::Params {
                dir,
                prompt,
                max_new,
                temperature,
            })?;
        }
    }
    Ok(())
}
