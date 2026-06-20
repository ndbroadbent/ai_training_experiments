//! Read an Obsidian vault and strip it down to clean prose.
//!
//! We deliberately keep cleaning light: remove YAML frontmatter, flatten
//! wikilinks/embeds/markdown links to their visible text, and collapse runs of
//! blank lines. Headings, lists, and emphasis are left intact — they're part of
//! the writing style we want the model to learn.
//!
//! Two views of the vault come out of here:
//! - [`load_corpus`]: one big cleaned string, for language-model training.
//! - [`load_chunks`]: passages with provenance, for retrieval.

use anyhow::Result;
use regex::Regex;
use std::path::Path;
use walkdir::WalkDir;

/// Cleaned, concatenated training text plus a count of contributing files.
pub struct Corpus {
    pub text: String,
    pub files: usize,
}

/// A retrievable passage and where it came from.
pub struct Chunk {
    pub source: String,
    pub text: String,
}

/// Compiled regexes for markdown cleanup, built once and reused per file.
struct Cleaner {
    frontmatter: Regex,
    embed: Regex,
    wikilink_alias: Regex,
    wikilink: Regex,
    mdlink: Regex,
    br_tag: Regex,
    md_escape: Regex,
    many_newlines: Regex,
}

impl Cleaner {
    fn new() -> Result<Self> {
        Ok(Self {
            // (?s) lets `.` match newlines; \A anchors to the very start.
            frontmatter: Regex::new(r"(?s)\A---\r?\n.*?\r?\n---\r?\n")?,
            embed: Regex::new(r"!\[\[[^\]]*\]\]")?, // ![[image.png]] -> removed
            wikilink_alias: Regex::new(r"\[\[[^\]|]+\|([^\]]+)\]\]")?, // [[a|b]] -> b
            wikilink: Regex::new(r"\[\[([^\]]+)\]\]")?, // [[a]] -> a
            mdlink: Regex::new(r"\[([^\]]+)\]\([^)]*\)")?, // [text](url) -> text
            br_tag: Regex::new(r"(?i)<br\s*/?>")?, // <br> -> newline
            md_escape: Regex::new(r"\\([\\\[\]()*_\-.#`~>])")?, // \_ -> _
            many_newlines: Regex::new(r"\n{3,}")?,
        })
    }

    fn clean(&self, raw: &str) -> String {
        let s = self.frontmatter.replace(raw, "");
        let s = self.embed.replace_all(s.as_ref(), "");
        let s = self.wikilink_alias.replace_all(s.as_ref(), "$1");
        let s = self.wikilink.replace_all(s.as_ref(), "$1");
        let s = self.mdlink.replace_all(s.as_ref(), "$1");
        let s = self.br_tag.replace_all(s.as_ref(), "\n");
        let s = self.md_escape.replace_all(s.as_ref(), "$1");
        self.many_newlines
            .replace_all(s.as_ref(), "\n\n")
            .trim()
            .to_string()
    }
}

/// Iterate every readable `.md` file under `root`, yielding (source, cleaned).
fn for_each_clean_file<F: FnMut(String, String)>(root: &Path, mut f: F) -> Result<()> {
    let cleaner = Cleaner::new()?;
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(path) else {
            continue; // skip unreadable / non-UTF8
        };
        let cleaned = cleaner.clean(&raw);
        if cleaned.is_empty() {
            continue;
        }
        let source = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        f(source, cleaned);
    }
    Ok(())
}

/// One big cleaned corpus, documents separated by a blank line.
pub fn load_corpus(root: &Path) -> Result<Corpus> {
    let mut text = String::new();
    let mut files = 0usize;
    for_each_clean_file(root, |_source, cleaned| {
        text.push_str(&cleaned);
        text.push_str("\n\n");
        files += 1;
    })?;
    Ok(Corpus { text, files })
}

/// Split the vault into retrievable passages of roughly `target_words` each.
/// Paragraphs are grouped together until the target is reached, so chunks stay
/// on topic and never split mid-paragraph.
pub fn load_chunks(root: &Path, target_words: usize) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    for_each_clean_file(root, |source, cleaned| {
        let mut buf = String::new();
        let mut words = 0usize;
        for para in cleaned.split("\n\n") {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }
            if !buf.is_empty() {
                buf.push_str("\n\n");
            }
            buf.push_str(para);
            words += para.split_whitespace().count();
            if words >= target_words {
                chunks.push(Chunk {
                    source: source.clone(),
                    text: std::mem::take(&mut buf),
                });
                words = 0;
            }
        }
        if !buf.trim().is_empty() {
            chunks.push(Chunk { source, text: buf });
        }
    })?;
    Ok(chunks)
}
