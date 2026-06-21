//! Generation "brain-scan": render an mp4 of the network firing as it emits
//! tokens for an `ask` query.
//!
//! For each generated token we capture the residual-stream activations after
//! every layer (via `Model::forward_capture`) and draw a frame: a layer×channel
//! heatmap, the text produced so far, and the top-k next-token distribution.
//! Frames are PNGs (pure Rust) encoded to mp4 with ffmpeg.

use crate::backend::{device, Raw};
use crate::data::format::{self, END};
use crate::data::{self, tokenizer::CharTokenizer};
use crate::model::{Model, ModelConfig};
use crate::retrieval::Bm25Index;
use crate::sample::pending_tool_call;
use crate::tools::calculator::{eval, format_number};
use ab_glyph::{Font, FontVec, ScaleFont};
use anyhow::{anyhow, bail, Result};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::softmax;
use image::{ImageFormat, Rgb, RgbImage};
use std::path::Path;
use std::process::Command;

const W: u32 = 1280;
const H: u32 = 720;
const BG: [u8; 3] = [16, 16, 22];

pub struct Params {
    pub dir: String,
    pub query: String,
    pub notes: String,
    pub retrieve: bool,
    pub top_k: usize,
    pub chunk_words: usize,
    pub max_new: usize,
    pub out: String,
    pub fps: usize,
}

pub fn run_viz(p: Params) -> Result<()> {
    let device = device();
    let config = ModelConfig::load(format!("{}/config.json", p.dir))
        .map_err(|e| anyhow!("loading config: {e:?}"))?;
    let tok = data::load_tokenizer(&format!("{}/tokenizer.json", p.dir))?;
    let model: Model<Raw> = config
        .init(&device)
        .load_file(format!("{}/model", p.dir), &CompactRecorder::new(), &device)
        .map_err(|e| anyhow!("loading model: {e:?}"))?;
    let font = load_font()?;

    // Build the prompt (optionally retrieval-grounded), same shape as `ask`.
    let context = if p.retrieve {
        let index = Bm25Index::from_vault(Path::new(&p.notes), p.chunk_words)?;
        index
            .search(&p.query, p.top_k)
            .iter()
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

    let frames_dir = "viz_frames";
    let _ = std::fs::remove_dir_all(frames_dir);
    std::fs::create_dir_all(frames_dir)?;

    let block = model.block_size();
    let vocab = tok.vocab_size();
    let layer_labels = layer_labels(config.n_layer);
    let encode = |s: &str| -> Vec<i32> { tok.encode(s).into_iter().map(|x| x as i32).collect() };

    let mut ids = encode(&prompt);
    let mut produced = String::new();
    let mut frame = 0usize;

    println!("Generating + rendering frames...");
    for step in 0..p.max_new {
        let ctx_len = ids.len().min(block);
        let ctx = ids[ids.len() - ctx_len..].to_vec();
        let x = Tensor::<Raw, 2, Int>::from_data(TensorData::new(ctx, [1, ctx_len]), &device);

        let (logits, acts) = model.forward_capture(x);
        let last = logits.slice([0..1, ctx_len - 1..ctx_len, 0..vocab]).reshape([vocab]);
        let probs: Vec<f32> = softmax(last, 0).into_data().to_vec().unwrap();
        let next = argmax(&probs);
        let top = top_tokens(&probs, &tok, 10);
        let ch = tok.decode(&[next as u32]);

        let img = compose(&font, &p.query, step, &acts, &layer_labels, &produced, &display(&ch), &top);
        img.save_with_format(format!("{frames_dir}/f_{frame:05}.png"), ImageFormat::Png)?;
        frame += 1;

        ids.push(next as i32);
        produced.push_str(&ch);

        if let Some(cut) = produced.find(END) {
            produced.truncate(cut);
            break;
        }
        // Resolve a completed tool call (injected text isn't model-emitted, so
        // no frame for it — it just appears in the text on the next step).
        if let Some(expr) = pending_tool_call(&produced) {
            let injected = match eval(&expr) {
                Ok(v) => format!("\n>>tool:result({})\n", format_number(v)),
                Err(_) => "\n>>tool:result(error)\n".to_string(),
            };
            ids.extend(encode(&injected));
            produced.push_str(&injected);
        }
    }

    // Hold the final frame for ~1.5s so the video ends on the full answer.
    if frame > 0 {
        let last = format!("{frames_dir}/f_{:05}.png", frame - 1);
        for _ in 0..(p.fps * 3 / 2) {
            std::fs::copy(&last, format!("{frames_dir}/f_{frame:05}.png"))?;
            frame += 1;
        }
    }

    println!("Encoding {frame} frames -> {} with ffmpeg...", p.out);
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            &p.fps.to_string(),
            "-i",
            &format!("{frames_dir}/f_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            &p.out,
        ])
        .status()?;
    if !status.success() {
        bail!("ffmpeg failed");
    }
    println!("Wrote {}", p.out);
    Ok(())
}

fn layer_labels(n_layer: usize) -> Vec<String> {
    let mut v = vec!["embed".to_string()];
    for i in 1..=n_layer {
        v.push(format!("L{i}"));
    }
    v.push("final".to_string());
    v
}

/// Compose a single frame.
#[allow(clippy::too_many_arguments)]
fn compose(
    font: &FontVec,
    query: &str,
    step: usize,
    acts: &[Vec<f32>],
    labels: &[String],
    produced: &str,
    next_ch: &str,
    top: &[(String, f32)],
) -> RgbImage {
    let mut img = RgbImage::from_pixel(W, H, Rgb(BG));

    // Title.
    draw_text(&mut img, font, &format!("ask:  {query}"), 20.0, 14.0, 26.0, [235, 235, 245]);
    draw_text(
        &mut img,
        font,
        &format!("step {step:>3}   emitting: \"{next_ch}\""),
        20.0,
        46.0,
        20.0,
        [150, 160, 200],
    );

    // Heatmap: rows = layers, columns = channels (residual stream activations).
    let (hx, hy, hw, rh) = (150i32, 84i32, 800i32, 34i32);
    draw_text(&mut img, font, "network activations (layer x channel)", hx as f32, 64.0, 18.0, [120, 130, 160]);
    for (li, row) in acts.iter().enumerate() {
        let scale = row.iter().fold(1e-6f32, |m, &v| m.max(v.abs()));
        let d = row.len().max(1);
        for px in 0..hw {
            let v = row[(px as usize * d / hw as usize).min(d - 1)] / scale;
            let color = heat(v);
            fill_rect(&mut img, hx + px, hy + li as i32 * rh, 1, rh - 2, color);
        }
        let ly = hy + li as i32 * rh;
        let label = labels.get(li).map(String::as_str).unwrap_or("");
        draw_text(&mut img, font, label, 70.0, ly as f32 - 2.0, 18.0, [180, 190, 210]);
    }

    // Top-k next-token distribution (right column).
    let (bx, by, bh) = (985i32, 84i32, 26i32);
    draw_text(&mut img, font, "next token", bx as f32, 64.0, 18.0, [120, 130, 160]);
    for (i, (s, prob)) in top.iter().enumerate() {
        let y = by + i as i32 * bh;
        let bar = (prob * 150.0) as i32;
        fill_rect(&mut img, bx + 70, y + 4, bar.max(1), bh - 10, [90, 140, 230]);
        draw_text(&mut img, font, &format!("{:>4.0}% {}", prob * 100.0, s), bx as f32, y as f32, 18.0, [210, 215, 230]);
    }

    // Generated text so far.
    draw_text(&mut img, font, "output:", 20.0, 392.0, 18.0, [120, 130, 160]);
    draw_multiline(&mut img, font, produced, 20.0, 416.0, 19.0, [120, 230, 150], 22.0, 12);

    img
}

fn display(ch: &str) -> String {
    match ch {
        "\n" => "\\n".into(),
        " " => "·".into(),
        other => other.into(),
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

fn top_tokens(probs: &[f32], tok: &CharTokenizer, k: usize) -> Vec<(String, f32)> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    idx.into_iter()
        .take(k)
        .map(|i| (display(&tok.decode(&[i as u32])), probs[i]))
        .collect()
}

/// Diverging colormap: warm = positive activation, cool = negative.
fn heat(v: f32) -> [u8; 3] {
    let v = v.clamp(-1.0, 1.0);
    if v >= 0.0 {
        let t = v.powf(0.6);
        [(35.0 + 220.0 * t) as u8, (20.0 + 170.0 * t * t) as u8, (50.0 * (1.0 - t)) as u8]
    } else {
        let t = (-v).powf(0.6);
        [(30.0 * (1.0 - t)) as u8, (70.0 + 110.0 * t) as u8, (70.0 + 185.0 * t) as u8]
    }
}

fn fill_rect(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, color: [u8; 3]) {
    for yy in y..y + h {
        for xx in x..x + w {
            if xx >= 0 && yy >= 0 && (xx as u32) < img.width() && (yy as u32) < img.height() {
                img.put_pixel(xx as u32, yy as u32, Rgb(color));
            }
        }
    }
}

fn blend(bg: u8, fg: u8, cov: f32) -> u8 {
    (bg as f32 * (1.0 - cov) + fg as f32 * cov).round() as u8
}

fn draw_text(img: &mut RgbImage, font: &FontVec, text: &str, x0: f32, y0: f32, px: f32, color: [u8; 3]) {
    let scaled = font.as_scaled(px);
    let ascent = scaled.ascent();
    let mut x = x0;
    for ch in text.chars() {
        let gid = font.glyph_id(ch);
        let glyph = gid.with_scale_and_position(px, ab_glyph::point(x, y0 + ascent));
        if let Some(outline) = font.outline_glyph(glyph) {
            let bb = outline.px_bounds();
            outline.draw(|gx, gy, cov| {
                let px_ = bb.min.x as i32 + gx as i32;
                let py_ = bb.min.y as i32 + gy as i32;
                if px_ >= 0 && py_ >= 0 && (px_ as u32) < img.width() && (py_ as u32) < img.height() {
                    let bg = img.get_pixel(px_ as u32, py_ as u32).0;
                    img.put_pixel(
                        px_ as u32,
                        py_ as u32,
                        Rgb([blend(bg[0], color[0], cov), blend(bg[1], color[1], cov), blend(bg[2], color[2], cov)]),
                    );
                }
            });
        }
        x += scaled.h_advance(gid);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_multiline(img: &mut RgbImage, font: &FontVec, text: &str, x0: f32, y0: f32, px: f32, color: [u8; 3], line_h: f32, max_lines: usize) {
    let lines: Vec<&str> = text.split('\n').collect();
    let start = lines.len().saturating_sub(max_lines);
    for (i, line) in lines[start..].iter().enumerate() {
        draw_text(img, font, line, x0, y0 + i as f32 * line_h, px, color);
    }
}

fn load_font() -> Result<FontVec> {
    let candidates = [
        "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ];
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(font) = FontVec::try_from_vec(bytes) {
                return Ok(font);
            }
        }
    }
    bail!("no usable TTF font found in known macOS font paths")
}
