//! The training-example schema and how it renders into one text sequence.
//!
//! An [`Example`] is the unit of supervised data. It renders into a flat string
//! with sentinel markers the model learns to recognize. The same markers are
//! used at inference to know where the answer starts and where tool calls live.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::Write;

// Sentinel markers. Chosen to be extremely unlikely to occur in real notes.
pub const CTX: &str = "<|context|>";
pub const QUESTION: &str = "<|question|>";
pub const ANSWER: &str = "<|answer|>";
pub const END: &str = "<|end|>";

// Tool-call convention (matches the user's `>>tool:calculate(...)` spec).
// Used by the inference tool-loop (task 6).
#[allow(dead_code)]
pub const TOOL_CALL_OPEN: &str = ">>tool:calculate(";
#[allow(dead_code)]
pub const TOOL_RESULT_OPEN: &str = ">>tool:result(";

/// One supervised example. `response` may already contain resolved tool calls
/// (`>>tool:calculate(...)` followed by `>>tool:result(...)`) inline.
#[derive(Serialize, Deserialize, Clone)]
pub struct Example {
    #[serde(default)]
    pub context: String,
    pub question: String,
    pub response: String,
    #[serde(default)]
    pub source: String,
}

impl Example {
    /// Render to the flat training sequence. Context is omitted when empty.
    pub fn render(&self) -> String {
        let mut s = String::new();
        if !self.context.trim().is_empty() {
            s.push_str(CTX);
            s.push('\n');
            s.push_str(self.context.trim());
            s.push('\n');
        }
        s.push_str(QUESTION);
        s.push('\n');
        s.push_str(self.question.trim());
        s.push('\n');
        s.push_str(ANSWER);
        s.push('\n');
        s.push_str(self.response.trim());
        s.push('\n');
        s.push_str(END);
        s.push('\n');
        s
    }
}

pub fn write_jsonl(path: &str, examples: &[Example]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for e in examples {
        writeln!(f, "{}", serde_json::to_string(e)?)?;
    }
    Ok(())
}

pub fn read_jsonl(path: &str) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if !line.is_empty() {
            out.push(serde_json::from_str(line)?);
        }
    }
    Ok(out)
}
