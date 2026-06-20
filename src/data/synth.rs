//! Synthetic training-data generators (the programmatic half of the hybrid).
//!
//! - [`gen_calculator`]: random arithmetic -> a resolved tool-call trace. Teaches
//!   the *format* and *when* to call the calculator (it never has to do math).
//! - [`gen_cloze`]: fill-in-the-blank from real note passages. Teaches grounded,
//!   extractive answering — find the fact in the provided context and copy it.
//!
//! These are mixed with the hand-authored seed set (see `data/seed_examples.jsonl`).

use crate::data::corpus::{self, Chunk};
use crate::data::format::{self, Example};
use crate::tools::calculator::{eval, format_number};
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use regex::Regex;
use std::path::Path;

/// Random, correctly-resolved calculator traces.
pub fn gen_calculator(n: usize, rng: &mut impl Rng) -> Vec<Example> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let op = ['+', '-', '*', '/'][rng.gen_range(0..4)];
        let (a, b) = match op {
            '*' => (rng.gen_range(11..1000), rng.gen_range(11..1000)),
            // Generate exact divisions so the answer is a clean integer.
            '/' => {
                let divisor = rng.gen_range(2..50);
                let quotient = rng.gen_range(2..200);
                (divisor * quotient, divisor)
            }
            _ => (rng.gen_range(10..100_000), rng.gen_range(10..100_000)),
        };
        let expr = format!("{a}{op}{b}");
        let result = format_number(eval(&expr).expect("generated expr is valid"));
        let word = match op {
            '+' => "plus",
            '-' => "minus",
            '*' => "times",
            _ => "divided by",
        };
        let templates = [
            format!("What is {a} {word} {b}?"),
            format!("Compute {expr}."),
            format!("What's {a} {word} {b}?"),
            format!("Calculate {a} {word} {b}."),
        ];
        let question = templates[rng.gen_range(0..templates.len())].clone();
        let response =
            format!(">>tool:calculate({expr})\n>>tool:result({result})\nThe answer is {result}.");
        out.push(Example {
            context: String::new(),
            question,
            response,
            source: "calc".into(),
        });
    }
    out
}

/// Find the sentence in `text` that contains `span` and has enough words.
fn sentence_with<'a>(text: &'a str, span: &str) -> Option<&'a str> {
    text.split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(str::trim)
        .find(|s| s.contains(span) && s.split_whitespace().count() >= 6)
}

/// Fill-in-the-blank QA grounded in real note passages. The answer (a number)
/// is present in the provided context, so the task is to locate and copy it.
pub fn gen_cloze(chunks: &[Chunk], n: usize, rng: &mut impl Rng) -> Vec<Example> {
    let num_re = Regex::new(r"\b\d[\d,]*\b").unwrap();
    let mut order: Vec<usize> = (0..chunks.len()).collect();
    order.shuffle(rng);

    let mut out = Vec::new();
    for &i in &order {
        if out.len() >= n {
            break;
        }
        let chunk = &chunks[i];
        for m in num_re.find_iter(&chunk.text) {
            let span = m.as_str();
            if span.chars().filter(char::is_ascii_digit).count() < 2 {
                continue; // need a non-trivial number
            }
            if let Some(sentence) = sentence_with(&chunk.text, span) {
                let blanked = sentence.replacen(span, "____", 1);
                if blanked != sentence {
                    out.push(Example {
                        context: chunk.text.clone(),
                        question: format!("Fill in the blank using the context:\n{blanked}"),
                        response: span.to_string(),
                        source: format!("cloze:{}", chunk.source),
                    });
                    break;
                }
            }
        }
    }
    out
}

pub struct Params {
    pub notes: String,
    pub out: String,
    pub calc: usize,
    pub cloze: usize,
    pub chunk_words: usize,
    pub seed: String,
    pub seed_repeat: usize,
}

/// `gen-data` subcommand: build the full mixed dataset and write train/val JSONL.
pub fn run_gen_data(p: Params) -> Result<()> {
    let mut rng = rand::thread_rng();
    let mut examples: Vec<Example> = Vec::new();

    // (1) Hand-authored seed set ("distilled by Claude"), upweighted because it
    //     is high quality and covers behaviors the generators don't (abstention,
    //     multi-step reasoning, RAG+tool combos).
    if Path::new(&p.seed).exists() {
        let seed = format::read_jsonl(&p.seed)?;
        println!(
            "Seed examples: {} x{} = {}",
            seed.len(),
            p.seed_repeat,
            seed.len() * p.seed_repeat
        );
        for _ in 0..p.seed_repeat {
            examples.extend(seed.iter().cloned());
        }
    } else {
        println!("(no seed file at {}, skipping)", p.seed);
    }

    // (2) Programmatic calculator traces.
    let calc = gen_calculator(p.calc, &mut rng);
    println!("Calculator examples: {}", calc.len());
    examples.extend(calc);

    // (3) Programmatic cloze QA grounded in the vault.
    if p.cloze > 0 {
        let chunks = corpus::load_chunks(Path::new(&p.notes), p.chunk_words)?;
        let cloze = gen_cloze(&chunks, p.cloze, &mut rng);
        println!("Cloze QA examples: {} (from {} chunks)", cloze.len(), chunks.len());
        examples.extend(cloze);
    }

    examples.shuffle(&mut rng);

    std::fs::create_dir_all(&p.out)?;
    let val_n = ((examples.len() as f64) * 0.05) as usize;
    let (val, train) = examples.split_at(val_n);
    format::write_jsonl(&format!("{}/train.jsonl", p.out), train)?;
    format::write_jsonl(&format!("{}/val.jsonl", p.out), val)?;
    println!(
        "\nWrote {} train + {} val examples to {}/",
        train.len(),
        val.len(),
        p.out
    );

    println!("\n=== sample rendered examples ===");
    for e in train.iter().take(3) {
        println!("{}----", e.render());
    }
    Ok(())
}
