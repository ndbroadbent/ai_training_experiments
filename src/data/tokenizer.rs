//! Character-level tokenizer.
//!
//! We start at the character level on purpose: zero dependencies, a tiny vocab,
//! and nothing to train. It's the simplest thing that works end-to-end. Once the
//! pipeline is proven we can swap in BPE (fewer, longer tokens) behind the same
//! `encode` / `decode` interface.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
pub struct CharTokenizer {
    /// id -> char (the source of truth; serialized).
    itos: Vec<char>,
    /// char -> id (derived from `itos`; rebuilt after load).
    #[serde(skip)]
    stoi: BTreeMap<char, u32>,
}

impl CharTokenizer {
    /// Build a vocab from every distinct character in `text` (sorted for a
    /// stable, reproducible id assignment).
    pub fn build(text: &str) -> Self {
        let set: std::collections::BTreeSet<char> = text.chars().collect();
        let itos: Vec<char> = set.into_iter().collect();
        let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
        Self { itos, stoi }
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    /// Single-character lookup, for encoding while keeping per-char alignment
    /// (e.g. with a loss mask). Returns `None` for out-of-vocab characters.
    pub fn id(&self, c: char) -> Option<u32> {
        self.stoi.get(&c).copied()
    }

    /// Encode text to token ids. Characters outside the vocab are dropped.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        s.chars().filter_map(|c| self.stoi.get(&c).copied()).collect()
    }

    /// Decode token ids back to text. Out-of-range ids are dropped.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().filter_map(|&i| self.itos.get(i as usize)).collect()
    }

    /// Rebuild the `stoi` map after deserializing (it is not serialized).
    #[allow(dead_code)] // used once tokenizer save/load lands (baseline training)
    pub fn rebuild(&mut self) {
        self.stoi = self
            .itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
    }
}
