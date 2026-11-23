//! Character-level language model built from scratch.
//!
//! Architecture:
//!   Embedding(vocab, embed_dim)
//!   Linear(embed_dim * context, hidden) -> ReLU
//!   Linear(hidden, vocab) -> Softmax
//!
//! Given `context` previous characters, predicts the next character.
//! Trained entirely from scratch with backpropagation — no pre-trained
//! weights, no external model files.

use crate::layers::{Embedding, Init, Linear};
use crate::optimizer::{Adam, SGD};
use crate::tensor::Tensor2;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub context: usize,
    pub init: Init,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 65,   // common ASCII-ish vocab
            embed_dim: 16,
            hidden_dim: 64,
            context: 8,
            init: Init::Xavier,
        }
    }
}

/// The from-scratch neural network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharLM {
    pub config: ModelConfig,
    pub embed: Embedding,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl CharLM {
    pub fn new(config: ModelConfig) -> Self {
        let init = config.init;
        Self {
            embed: Embedding::new(config.vocab_size, config.embed_dim, init),
            fc1: Linear::new(config.embed_dim * config.context, config.hidden_dim, init),
            fc2: Linear::new(config.hidden_dim, config.vocab_size, init),
            config,
        }
    }

    /// Forward pass. `indices` is a slice of `context` character ids.
    /// Returns logits (batch=1, vocab_size).
    pub fn forward(&mut self, indices: &[usize]) -> Tensor2 {
        assert_eq!(indices.len(), self.config.context);

        // Embed each char -> (context, embed_dim)
        let emb = self.embed.forward(indices);

        // Flatten to (1, context * embed_dim)
        let (c, d) = emb.shape();
        let mut flat = Tensor2::zeros(1, c * d);
        for i in 0..c {
            for j in 0..d {
                flat.data[[0, i * d + j]] = emb.data[[i, j]];
            }
        }

        let h = self.fc1.forward(&flat);
        let h = h.relu();
        let logits = self.fc2.forward(&h);
        logits
    }

    /// Predict next-char probabilities for a context.
    pub fn predict(&mut self, indices: &[usize]) -> Vec<f32> {
        let logits = self.forward(indices);
        let probs = logits.softmax();
        probs.data.row(0).to_owned().to_vec()
    }

    /// Sample a character index from probabilities (temperature scaling).
    pub fn sample(probs: &[f32], temperature: f32) -> usize {
        let mut scaled: Vec<f32> = probs.iter().map(|&p| {
            if p <= 0.0 {
                f32::NEG_INFINITY
            } else {
                (p.ln() / temperature).exp()
            }
        }).collect();
        let sum: f32 = scaled.iter().sum();
        if sum == 0.0 || !sum.is_finite() {
            scaled = probs.to_vec();
        }
        let total: f32 = scaled.iter().sum();
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen_range(0.0..total);
        let mut acc = 0.0;
        for (i, &p) in scaled.iter().enumerate() {
            acc += p;
            if r <= acc {
                return i;
            }
        }
        probs.len().saturating_sub(1)
    }

    /// Zero all gradients.
    pub fn zero_grads(&mut self) {
        self.embed.zero_grads();
        self.fc1.zero_grads();
        self.fc2.zero_grads();
    }

    /// Apply SGD updates to all params.
    pub fn apply_sgd(&mut self, lr: f32) {
        self.embed.apply_grad(lr);
        self.fc1.apply_grad(lr);
        self.fc2.apply_grad(lr);
    }

    /// Save model to JSON file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model from JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model: Self = serde_json::from_str(&json)?;
        Ok(model)
    }
}

/// Trainer: runs the training loop.
pub struct Trainer {
    pub model: CharLM,
    pub lr: f32,
}

impl Trainer {
    pub fn new(model: CharLM, _optimizer: String, lr: f32) -> Self {
        // SGD via apply_sgd is used in train_step. Adam can be added later.
        let _ = Adam::default();
        let _ = SGD::default();
        Self { model, lr }
    }

    /// Train on one mini-batch of (context_indices, target_index).
    pub fn train_step(&mut self, batches: &[(Vec<usize>, usize)]) -> f32 {
        self.model.zero_grads();
        let mut total_loss = 0.0;
        let n = batches.len();

        for (indices, target) in batches.iter() {
            // Forward
            let logits = self.model.forward(indices);
            let targets = vec![*target];

            // Loss
            let loss = logits.cross_entropy(&targets);
            total_loss += loss;

            // Backprop through fc2: dL/dlogits
            let grad_logits = logits.softmax_cross_entropy_grad(&targets);

            // Recompute fc1 activations so we have the post-relu output cached.
            let emb = self.model.embed.forward(indices);
            let (c, d) = emb.shape();
            let mut flat = Tensor2::zeros(1, c * d);
            for i in 0..c {
                for j in 0..d {
                    flat.data[[0, i * d + j]] = emb.data[[i, j]];
                }
            }
            let h_pre = self.model.fc1.forward(&flat);

            // fc2 backward using h as its input
            let grad_h = self.model.fc2.backward(&grad_logits);
            // grad_h is (1, hidden). Apply relu derivative.
            let grad_h_pre = Tensor2 {
                data: grad_h.data * &h_pre.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            };

            // fc1 backward: input is flat
            let grad_flat = self.model.fc1.backward(&grad_h_pre);

            // Embedding backward: grad_flat is (1, c*d). Split back to (c, d).
            let mut grad_emb = Tensor2::zeros(c, d);
            for i in 0..c {
                for j in 0..d {
                    grad_emb.data[[i, j]] = grad_flat.data[[0, i * d + j]];
                }
            }
            self.model.embed.backward(&grad_emb);
        }

        // Apply SGD updates (gradients were summed over the batch, so scale lr)
        self.model.apply_sgd(self.lr / n.max(1) as f32);
        total_loss / n.max(1) as f32
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        self.model.save(path)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let model = CharLM::load(path)?;
        Ok(Self { model, lr: 0.01 })
    }
}

/// Character-level tokenizer: maps chars <-> ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub chars: Vec<char>,
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: HashMap<usize, char>,
}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i);
            id_to_char.insert(i, c);
        }
        Self { chars, char_to_id, id_to_char }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| self.char_to_id.get(&c).copied().unwrap_or(0)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().filter_map(|&i| self.id_to_char.get(&i).copied()).collect()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_forward_shape() {
        let config = ModelConfig { vocab_size: 10, embed_dim: 4, hidden_dim: 8, context: 3, init: Init::Xavier };
        let mut model = CharLM::new(config);
        let logits = model.forward(&[0, 1, 2]);
        assert_eq!(logits.shape(), (1, 10));
    }

    #[test]
    fn tokenizer_roundtrip() {
        let tok = Tokenizer::from_text("hello world");
        let ids = tok.encode("hello");
        let dec = tok.decode(&ids);
        assert_eq!(dec, "hello");
    }

    #[test]
    fn train_step_runs() {
        let config = ModelConfig { vocab_size: 5, embed_dim: 4, hidden_dim: 8, context: 3, init: Init::Xavier };
        let model = CharLM::new(config);
        let mut trainer = Trainer::new(model, "sgd".into(), 0.1);
        let batches = vec![
            (vec![0, 1, 2], 3),
            (vec![1, 2, 3], 4),
        ];
        let loss = trainer.train_step(&batches);
        assert!(loss.is_finite());
    }
}
