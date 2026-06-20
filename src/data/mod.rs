//! Data pipeline: turn a messy Obsidian vault into clean training text and a
//! tokenizer. This is the front of the whole project — everything trains on
//! what comes out of here.

pub mod corpus;
pub mod format;
pub mod synth;
pub mod tokenizer;

use anyhow::Result;
use std::path::Path;
use std::path::PathBuf;
use tokenizer::CharTokenizer;

/// `data` subcommand: load + clean the vault, build a char tokenizer, report.
pub fn run_data(notes: PathBuf) -> Result<()> {
    println!("Loading corpus from {}", notes.display());
    let corpus = corpus::load_corpus(&notes)?;
    let char_count = corpus.text.chars().count();

    println!("Files used:         {}", corpus.files);
    println!("Characters (clean): {char_count}");

    let tok = CharTokenizer::build(&corpus.text);
    println!("Vocab size:         {}", tok.vocab_size());

    let sample: String = corpus.text.chars().take(400).collect();
    println!("\n--- sample (first 400 clean chars) ---\n{sample}\n--- end sample ---");

    let ids = tok.encode(&sample);
    let back = tok.decode(&ids);
    println!("\nTokenizer round-trip ok: {}", back == sample);

    Ok(())
}

/// Load the vault, build a tokenizer, and return the whole corpus as token ids.
pub fn load_tokens(notes: &Path) -> Result<(Vec<i32>, CharTokenizer)> {
    let corpus = corpus::load_corpus(notes)?;
    let tok = CharTokenizer::build(&corpus.text);
    let ids: Vec<i32> = tok.encode(&corpus.text).into_iter().map(|x| x as i32).collect();
    Ok((ids, tok))
}

pub fn save_tokenizer(tok: &CharTokenizer, path: &str) -> Result<()> {
    std::fs::write(path, serde_json::to_string(tok)?)?;
    Ok(())
}

pub fn load_tokenizer(path: &str) -> Result<CharTokenizer> {
    let mut tok: CharTokenizer = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    tok.rebuild(); // stoi is not serialized; reconstruct it from itos
    Ok(tok)
}
