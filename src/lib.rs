//! # cactus-rs
//!
//! A from-scratch neural network in Rust. Trains its own model — no pre-trained
//! weights, no GGUF, no external model files. Smallest binary, fastest CPU
//! training/inference, easy integration for any developer.
//!
//! Includes:
//! - `model`  — character-level language model (text)
//! - `turbo`  — compact 100M-shape low-rank language model (<1 MB on disk)
//! - `web`    — zero-dependency HTTP API + browser UI
//! - `vision` — convolutional neural network (images)
//!
//! ## Quick start (developer)
//!
//! ```no_run
//! use cactus_rs::{Tokenizer, CharLM, Trainer};
//!
//! # fn main() -> anyhow::Result<()> {
//! let text = "hello world";
//! let tok = Tokenizer::from_text(text);
//! let config = cactus_rs::ModelConfig {
//!     vocab_size: tok.vocab_size(),
//!     ..Default::default()
//! };
//! let model = CharLM::new(config);
//! let mut trainer = Trainer::new(model, "sgd".into(), 0.1);
//!
//! let ids = tok.encode(text);
//! let mut batches = Vec::new();
//! for i in 0..ids.len().saturating_sub(trainer.model.config.context) {
//!     let ctx: Vec<usize> = ids[i..i + trainer.model.config.context].to_vec();
//!     let target = ids[i + trainer.model.config.context];
//!     batches.push((ctx, target));
//! }
//! for _ in 0..10 {
//!     trainer.train_step(&batches);
//! }
//! # Ok(()) }
//! ```

pub mod layers;
pub mod model;
pub mod optimizer;
pub mod tensor;
pub mod turbo;
pub mod vision;
pub mod web;

pub use layers::{Embedding, Init, Linear};
pub use model::{CharLM, ModelConfig, Tokenizer, Trainer};
pub use optimizer::{Adam, SGD};
pub use tensor::{Tensor1, Tensor2};
pub use turbo::{generate_text, LowRankLinear, TurboConfig, TurboLM, TurboTrainer};
pub use vision::{load_image, load_manifest, Conv2d, MaxPool2d, Tensor4, VisionCNN, VisionTrainer};
