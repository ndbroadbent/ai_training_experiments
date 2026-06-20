# ai_training_experiments

Sandbox for training small language models **from scratch in Rust**
([Burn](https://burn.dev)) on my own Obsidian vault, running on a laptop GPU
(Apple Silicon / Metal).

## Philosophy: a "RISC" model

Most of a transformer's parameters store **facts** — its feed-forward layers act
as key-value memories. This project deliberately goes the other way:

> **Memorize skills and language; externalize facts.**

It's a *Reduced Instruction-Set Cognition* model: a small core trained to
**reason, call tools, and read retrieved context** rather than to recall facts
from its weights. Knowledge lives *outside* the model and is fetched on demand.
This mirrors real research — RETRO, Toolformer, kNN-LM, and the finding that
transformer feed-forward layers are key-value fact stores.

Two concrete embodiments:

- **Calculator tool.** The model emits `>>tool:calculate(2334*9834)`; the runtime
  evaluates it, injects `>>tool:result(22952556)`, and the model continues. It
  learns *when* to call the tool, never *how* to do arithmetic.
- **Retrieval (RAG).** Notes are chunked and searched with hand-rolled **BM25**;
  the top passages become the model's context. (Wikipedia is intended as external
  memory too — indexed, never trained on.)

## Results

Trained on **13.7M characters** (8,647 markdown notes, 975-char vocab) on an
Apple M2 Pro.

### Baseline transformer (the "memorize everything" control)

| | |
|---|---|
| Architecture | 6 layers, 6 heads, d_model 384, block 256, char-level |
| Parameters | 11.5M |
| Training | 2,000 steps, AdamW, Metal GPU |
| Final loss | train **1.64** / val **1.69** (~2.4 bits/char) |

Sample at val 1.69 — real words and local grammar, no global coherence (textbook
for a ~1.7-nats char model):

> *…a the cause fore probably go the the much the day project officing a the story
> one see and and new store one. They be operative dols or my balay…*

### RISC model (fine-tuned from the baseline)

Supervised fine-tuning on **6,270 synthetic examples** (programmatic calculator +
cloze-QA from the vault + a hand-authored seed set), with **answer-only loss
masking** — no loss on the prompt *or* on injected tool results, so the model is
never trained to compute. Masked answer loss collapsed to **~0.002 val**.

The calculator tool-loop, end to end:

```
$ ask --no-retrieve --query "What is 2334 times 9834?"
Q: What is 2334 times 9834?
A: >>tool:calculate(2334*9834)
   >>tool:result(22952556)        ← evaluated by the Rust evaluator, injected
   2334 times 9834 is 22952556.   ✓

$ ask --no-retrieve --query "What is 87654 minus 12345?"
A: >>tool:calculate(87654-12345)
   >>tool:result(75309)
   The answer is 75309.           ✓
```

### What works / what doesn't (honest scorecard)

| Capability | Status | Notes |
|---|---|---|
| Calculator tool-loop (detect → eval → inject → resume) | ✅ works | the whole mechanism is solid |
| Direct arithmetic (×, −, +, ÷, large numbers) | ✅ correct | matches trained templates, copies operands |
| BM25 retrieval (notes → top-k passages) | ✅ works | precise; surfaces the right note |
| Multi-step word problems | ❌ wrong expr | too few examples → no generalization |
| Grounded retrieval QA | ❌ weak | train/test mismatch + thin data + 256-char window |

Every *engineering* piece works (masked SFT, tool-loop, retrieval, format). The
failures are **data + scale**, not architecture — fixable with more diverse
training data and a larger context window (see Roadmap).

## How it works

- **Backends** — CPU (`ndarray`) by default; Metal GPU (`wgpu`) behind a `gpu`
  cargo feature. The same code trains and infers on either.
- **Tokenizer** — character-level (zero-dependency, represents every marker and
  tool call). BPE is a planned drop-in upgrade behind the same interface.
- **Data pipeline** — cleans the vault (frontmatter, wikilinks, `<br>`, escapes),
  builds either one corpus (for pretraining) or chunks with provenance (for RAG).
- **SFT format** — examples render to a flat sequence with sentinel markers
  (`<|context|> <|question|> <|answer|> <|end|>`) and inline tool calls.

## Project layout

```
src/
  main.rs         CLI (hello, data, gen-data, retrieve, train, train-sft, ask, sample)
  backend.rs      CPU / Metal backend selection
  model.rs        decoder-only transformer (configurable FFN width)
  train.rs        baseline pretraining loop
  sft.rs          masked-loss supervised fine-tuning
  sample.rs       generation + the calculator tool-loop + `ask`
  retrieval.rs    hand-rolled BM25 index
  tools/calculator.rs   recursive-descent arithmetic evaluator (+ tests)
  data/           corpus cleaning, char tokenizer, example format, synth generators
data/             generated train/val JSONL + hand-authored seed_examples.jsonl
```

## Usage

```sh
# Backend + autodiff sanity check (add --features gpu for Metal)
cargo run -- hello

# Inspect the vault: clean it, build a char tokenizer, print stats
cargo run -- data

# BM25 retrieval over the vault
cargo run -- retrieve --query "chess opening fundamentals" --top-k 3

# Generate the synthetic SFT dataset (calculator + cloze + seed)
cargo run -- gen-data --calc 3000 --cloze 3000

# Pretrain the baseline transformer (GPU recommended)
cargo run --features gpu --release -- train --steps 2000 --out checkpoints/baseline

# Fine-tune the RISC model from the baseline
cargo run --features gpu --release -- train-sft --init checkpoints/baseline --out checkpoints/risc

# Ask a question: pure tool-use (no retrieval) ...
cargo run --features gpu --release -- ask --dir checkpoints/risc --no-retrieve \
  --query "What is 2334 times 9834?"

# ... or grounded in retrieved notes
cargo run --features gpu --release -- ask --dir checkpoints/risc --query "<question about your notes>"

# Free-form sampling from any checkpoint
cargo run --features gpu --release -- sample --dir checkpoints/baseline --prompt "Chess is"
```

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | Scaffold + Metal/CPU backends | ✅ |
| 2 | Data pipeline + char tokenizer | ✅ |
| 3 | Baseline transformer | ✅ |
| 4 | BM25 retrieval | ✅ |
| 5 | Synthetic data (calculator + cloze + seed) | ✅ |
| 6 | RISC model + calculator tool-loop | ✅ |
| 7 | Experimental N-D neuron-grid block | ⏳ planned |

**Next up:**
- **Better Q&A** — more diverse, natural-question grounded examples; retrain the
  baseline at block 512–1024 (the 256-char window is the main RAG bottleneck);
  optionally BPE + vocab pruning.
- **Task 7 — the neuron grid.** Replace the transformer's feed-forward block with
  a high-dimensional *Moore-connected lattice* (orthogonal + diagonal + cross-dim
  neighbors) whose cells update iteratively — a 3D→5D "grid-as-block" experiment,
  compared head-to-head against the baseline FFN at matched parameters.

## Requirements

Rust 1.89+, and an Apple Silicon Mac for the Metal backend (CPU works anywhere).
Built with Burn 0.20.1.
