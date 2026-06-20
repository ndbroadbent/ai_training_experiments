//! Lexical retrieval (BM25) over vault passages — the external "memory" the
//! RISC model will look facts up in, rather than storing them in its weights.
//!
//! Hand-rolled rather than using a search-engine crate: it's small, transparent,
//! has zero heavy dependencies, and builds an in-memory index over the notes in
//! well under a second. When we add the Wikipedia subset we can swap in an
//! on-disk index (e.g. tantivy) behind the same `search` interface.

use crate::data::corpus::{self, Chunk};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

const K1: f32 = 1.5; // term-frequency saturation
const B: f32 = 0.75; // length-normalization strength

/// Lowercase and split on non-alphanumeric boundaries. Simple but effective;
/// no stemming (a deliberate toy-scale choice).
fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

struct Doc {
    source: String,
    text: String,
    len: u32,
}

pub struct SearchResult {
    pub score: f32,
    pub source: String,
    pub snippet: String,
}

pub struct Bm25Index {
    docs: Vec<Doc>,
    /// term -> [(doc_id, term_frequency)]
    postings: HashMap<String, Vec<(usize, u32)>>,
    avgdl: f32,
}

impl Bm25Index {
    /// Build an in-memory BM25 index from passages.
    pub fn build(chunks: Vec<Chunk>) -> Self {
        let mut docs = Vec::with_capacity(chunks.len());
        let mut postings: HashMap<String, Vec<(usize, u32)>> = HashMap::new();
        let mut total_len: u64 = 0;

        for (doc_id, chunk) in chunks.into_iter().enumerate() {
            let tokens = tokenize(&chunk.text);
            let mut tf: HashMap<&str, u32> = HashMap::new();
            for t in &tokens {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            for (term, count) in tf {
                postings.entry(term.to_string()).or_default().push((doc_id, count));
            }
            total_len += tokens.len() as u64;
            docs.push(Doc {
                source: chunk.source,
                text: chunk.text,
                len: tokens.len() as u32,
            });
        }

        let avgdl = if docs.is_empty() {
            0.0
        } else {
            total_len as f32 / docs.len() as f32
        };
        Self { docs, postings, avgdl }
    }

    /// Build directly from a notes vault.
    pub fn from_vault(root: &Path, chunk_words: usize) -> Result<Self> {
        Ok(Self::build(corpus::load_chunks(root, chunk_words)?))
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Return the top-`k` passages for `query`, scored by BM25.
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let n = self.docs.len() as f32;
        let mut query_terms = tokenize(query);
        query_terms.sort();
        query_terms.dedup();

        let mut scores: HashMap<usize, f32> = HashMap::new();
        for term in &query_terms {
            let Some(plist) = self.postings.get(term) else {
                continue;
            };
            let df = plist.len() as f32;
            // Robertson/Sparck-Jones idf with the standard +0.5 smoothing.
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for &(doc_id, tf) in plist {
                let dl = self.docs[doc_id].len as f32;
                let tf = tf as f32;
                let denom = tf + K1 * (1.0 - B + B * dl / self.avgdl);
                *scores.entry(doc_id).or_insert(0.0) += idf * (tf * (K1 + 1.0)) / denom;
            }
        }

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(k);

        ranked
            .into_iter()
            .map(|(doc_id, score)| {
                let doc = &self.docs[doc_id];
                SearchResult {
                    score,
                    source: doc.source.clone(),
                    snippet: snippet(&doc.text, 240),
                }
            })
            .collect()
    }
}

/// First `max_chars` characters of `text`, single-lined for display.
fn snippet(text: &str, max_chars: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let s: String = flat.chars().take(max_chars).collect();
    if flat.chars().count() > max_chars {
        format!("{s}…")
    } else {
        s
    }
}

pub struct Params {
    pub notes: String,
    pub query: String,
    pub top_k: usize,
    pub chunk_words: usize,
}

/// `retrieve` subcommand: build the index and run one query.
pub fn run_retrieve(p: Params) -> Result<()> {
    println!("Indexing {} ...", p.notes);
    let index = Bm25Index::from_vault(Path::new(&p.notes), p.chunk_words)?;
    println!("Indexed {} passages.\n", index.len());

    println!("Query: {:?}\n", p.query);
    let results = index.search(&p.query, p.top_k);
    if results.is_empty() {
        println!("(no matches)");
    }
    for (i, r) in results.iter().enumerate() {
        println!("#{}  score {:.3}  [{}]", i + 1, r.score, r.source);
        println!("    {}\n", r.snippet);
    }
    Ok(())
}
