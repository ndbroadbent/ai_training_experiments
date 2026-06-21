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

> **Reality check learned the hard way:** you can't externalize *everything*.
> Reasoning, grammar, and language understanding have to live in the weights —
> the model can't even parse a question without them. "Externalize facts" works;
> "externalize knowledge" doesn't, because reasoning rides on absorbed knowledge.

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
```

| Capability | Status | Notes |
|---|---|---|
| Calculator tool-loop (detect → eval → inject → resume) | ✅ works | the whole mechanism is solid |
| Direct arithmetic (×, −, +, ÷, large numbers) | ✅ correct | matches trained templates, copies operands |
| BM25 retrieval (notes → top-k passages) | ✅ works | precise; surfaces the right note |
| Multi-step word problems | ❌ wrong expr | too few examples → no generalization |
| Grounded retrieval QA | ❌ weak | train/test mismatch + thin data + 256-char window |

Every *engineering* piece works (masked SFT, tool-loop, retrieval, format). The
failures are **data + scale**, not architecture.

### Experimental 5D neuron-grid (vs. the MLP)

Each transformer block's feed-forward layer can be swapped (`--use-grid`) for an
**N-dimensional Moore-connected lattice**: a token's vector is projected into a
grid of cells that update from their neighbors — orthogonal, diagonal, and
cross-dimension (diagonals give Chebyshev-distance mixing, so signal spreads
fast) — for several iterations, then projected back. The topology lives in a
constant adjacency matrix, so 3D→5D is just a flag.

First run: a 5D grid (243 cells × 6 channels, 3 iterations), parameter-matched to
the MLP (11.13M vs 11.49M), same 6L/384/256 transformer, 1,000 steps:

| step | baseline MLP (val) | 5D-grid (val) |
|---|---|---|
| 200 | 2.63 | 2.64 |
| 600 | 2.33 | 2.47 |
| 1000 | **2.00** | **2.15** |

**Takeaway:** the grid learns language at near-MLP rate and the gap *stabilizes*
at ~0.15 nats — a hand-built neuron lattice is in the same league as the FFN that
powers every production transformer, at equal parameters. It doesn't beat it
(and underfits, so there's headroom). Caveat: `side=3` is the *dense-mixing*
regime (every cell neighbors every other), so this run doesn't yet test true
locality (`side≥4`) — the most interesting part of the idea.

## Visualization

`viz` renders an **mp4 of the network firing as it answers a query**: per emitted
token it captures the residual-stream activations after every layer and draws a
frame — a layer×channel heatmap, the live output text (including the tool-loop),
and the top-k next-token distribution. Pure-Rust frames (`image` + `ab_glyph`),
encoded to mp4 with ffmpeg.

```sh
cargo run -- viz --dir checkpoints/risc --no-retrieve \
  --query "What is 2334 times 9834?" --out viz.mp4
```

## How it works

- **Backends** — CPU (`ndarray`) by default; Metal GPU (`wgpu`) behind a `gpu`
  cargo feature. The same code trains and infers on either.
- **Tokenizer** — character-level (zero-dependency, represents every marker and
  tool call). BPE is a planned drop-in upgrade behind the same interface.
- **Data pipeline** — cleans the vault (frontmatter, wikilinks, `<br>`, escapes),
  builds either one corpus (for pretraining) or chunks with provenance (for RAG).
- **SFT format** — examples render to a flat sequence with sentinel markers
  (`<|context|> <|question|> <|answer|> <|end|>`) and inline tool calls.
- **Grid block** — a drop-in feed-forward replacement; dimensionality, side,
  channels, and iterations are all hyperparameters.

## Project layout

```
src/
  main.rs         CLI (hello, data, gen-data, retrieve, train, train-sft, ask, viz, sample)
  backend.rs      CPU / Metal backend selection
  model.rs        decoder-only transformer (MLP or N-D grid feed-forward)
  train.rs        baseline pretraining loop
  sft.rs          masked-loss supervised fine-tuning
  sample.rs       generation + the calculator tool-loop + `ask`
  viz.rs          mp4 "brain-scan" of the net firing during generation
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

# ... or pretrain with the experimental 5D neuron-grid feed-forward
cargo run --features gpu --release -- train --use-grid --grid-dims 5 --grid-side 3 \
  --grid-channels 6 --grid-iters 3 --out checkpoints/grid5d

# Fine-tune the RISC model from the baseline
cargo run --features gpu --release -- train-sft --init checkpoints/baseline --out checkpoints/risc

# Ask a question: pure tool-use (no retrieval) ...
cargo run --features gpu --release -- ask --dir checkpoints/risc --no-retrieve \
  --query "What is 2334 times 9834?"

# ... or grounded in retrieved notes
cargo run --features gpu --release -- ask --dir checkpoints/risc --query "<question about your notes>"

# Visualize the network firing as it answers (-> viz.mp4)
cargo run -- viz --dir checkpoints/risc --no-retrieve --query "What is 2334 times 9834?"

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
| 7 | Experimental N-D neuron-grid block | ✅ |
| + | Generation visualizer (mp4) | ✅ |

**Open follow-ups:**
- **Grid locality** — a `side≥4` run (true Moore neighborhoods, not dense mixing)
  and a sweep of `grid-iters` / `grid-channels` to see if either closes the
  ~0.15-nat gap to the MLP.
- **Better Q&A** — more diverse, natural-question grounded examples; retrain the
  baseline at block 512–1024 (the 256-char window is the main RAG bottleneck);
  optionally BPE + vocab pruning.

## Requirements

Rust 1.89+, an Apple Silicon Mac for the Metal backend (CPU works anywhere), and
`ffmpeg` for the `viz` command. Built with Burn 0.20.1.
