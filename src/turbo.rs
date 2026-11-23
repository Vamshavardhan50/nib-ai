//! Turbo: a ~100M logical-parameter model stored in under 1 MB.
//!
//! A dense 100M-parameter model needs ~400 MB at f32 or ~12.5 MB even at 1 bit,
//! so "100M independent parameters in <1 MB" is physically impossible. This
//! module instead builds a model whose **dense (logical) parameter shape** is
//! ~100M but whose **stored** parameter count is tiny, via:
//!
//!   1. Low-rank factorization: W (in × out) = U (in × r) · V (r × out).
//!   2. int8 quantization on save.
//!
//! Example: context=64, embed=128 -> in=8192; hidden=12114; rank=30;
//! logical params ≈ 100M; stored ≈ 995K values -> ~0.995 MB as int8.

use crate::layers::{Embedding, Init};
use crate::tensor::{Tensor1, Tensor2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"CTURBO01";

/// Low-rank linear layer: y = x @ U @ V + b.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRankLinear {
    pub u: Tensor2, // (in_features, rank)
    pub v: Tensor2, // (rank, out_features)
    pub bias: Tensor1,
    pub d_u: Tensor2,
    pub d_v: Tensor2,
    pub d_bias: Tensor1,
    last_input: Option<Tensor2>,
    last_z: Option<Tensor2>,
}

impl LowRankLinear {
    pub fn new(in_features: usize, out_features: usize, rank: usize, init: Init) -> Self {
        // Initialize the *product* U·V with the requested variance. Sampling
        // both factors with the dense-layer scale would make the product far
        // too small (scale^2) and the network would barely train.
        let (u_scale, v_scale) = match init {
            Init::Xavier => {
                let target_var = 2.0 / (in_features + out_features) as f32;
                (
                    (3.0 / in_features as f32).sqrt(),
                    (3.0 * target_var * in_features as f32 / rank as f32).sqrt(),
                )
            }
            Init::He => (
                (3.0 / in_features as f32).sqrt(),
                (6.0 / rank as f32).sqrt(),
            ),
            Init::Lecun => (
                (3.0 / in_features as f32).sqrt(),
                (3.0 / rank as f32).sqrt(),
            ),
            Init::Constant(c) => (c, (1.0 / rank as f32).sqrt()),
        };
        Self {
            u: Tensor2::random(in_features, rank, u_scale),
            v: Tensor2::random(rank, out_features, v_scale),
            bias: Tensor1::zeros(out_features),
            d_u: Tensor2::zeros(in_features, rank),
            d_v: Tensor2::zeros(rank, out_features),
            d_bias: Tensor1::zeros(out_features),
            last_input: None,
            last_z: None,
        }
    }

    pub fn forward(&mut self, x: &Tensor2) -> Tensor2 {
        let z = x.matmul(&self.u); // (batch, rank)
        let mut y = z.matmul(&self.v); // (batch, out)
        y = y.add(&bias_broadcast(&self.bias, y.rows()));
        self.last_input = Some(x.clone());
        self.last_z = Some(z);
        y
    }

    pub fn backward(&mut self, grad_y: &Tensor2) -> Tensor2 {
        let x = self.last_input.as_ref().expect("forward before backward");
        let z = self.last_z.as_ref().expect("forward before backward");

        // dV += z^T @ grad_y
        let dv = transpose(z).matmul(grad_y);
        self.d_v.add_scaled(&dv, 1.0);

        // db += sum rows
        let db = grad_y.sum_rows();
        self.d_bias.add_scaled(&db, 1.0);

        // dZ = grad_y @ V^T
        let dz = grad_y.matmul(&transpose(&self.v));

        // dU += x^T @ dZ
        let du = transpose(x).matmul(&dz);
        self.d_u.add_scaled(&du, 1.0);

        // dX = dZ @ U^T
        dz.matmul(&transpose(&self.u))
    }

    pub fn apply_grad(&mut self, lr: f32) {
        self.u.add_scaled(&self.d_u, -lr);
        self.v.add_scaled(&self.d_v, -lr);
        self.bias.add_scaled(&self.d_bias, -lr);
    }

    pub fn zero_grads(&mut self) {
        self.d_u.data.fill(0.0);
        self.d_v.data.fill(0.0);
        self.d_bias.data.fill(0.0);
    }

    /// Stored parameter count (U + V + bias).
    pub fn stored_params(&self) -> usize {
        self.u.rows() * self.u.cols() + self.v.rows() * self.v.cols() + self.bias.data.len()
    }

    /// Dense logical parameter count (in × out + bias).
    pub fn logical_params(&self) -> usize {
        self.u.rows() * self.v.cols() + self.bias.data.len()
    }
}

/// Configuration for the Turbo model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub context: usize,
    pub rank: usize,
}

impl Default for TurboConfig {
    fn default() -> Self {
        Self {
            vocab_size: 65,
            embed_dim: 128,
            hidden_dim: 12288,
            context: 64,
            rank: 16,
        }
    }
}

/// The Turbo character-level model (low-rank + quantizable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboLM {
    pub config: TurboConfig,
    pub embed: Embedding,
    pub fc1: LowRankLinear,
    pub fc2: LowRankLinear,
}

impl TurboLM {
    pub fn new(config: TurboConfig) -> Self {
        let in_features = config.embed_dim * config.context;
        let init = Init::Xavier;
        Self {
            embed: Embedding::new(config.vocab_size, config.embed_dim, init),
            fc1: LowRankLinear::new(in_features, config.hidden_dim, config.rank, init),
            fc2: LowRankLinear::new(config.hidden_dim, config.vocab_size, config.rank, init),
            config,
        }
    }

    /// Auto-size `hidden_dim` so the dense logical parameter count is close to
    /// `target`, while keeping the stored (quantized) size small.
    pub fn with_logical_target(mut self, target: usize) -> Self {
        let v = self.config.vocab_size;
        let e = self.config.embed_dim;
        let c = self.config.context;
        let denom = c * e + v;
        let h = target.saturating_sub(v * e) / denom.max(1);
        self.config.hidden_dim = h.max(64);
        let in_features = c * e;
        let init = Init::Xavier;
        self.fc1 = LowRankLinear::new(in_features, self.config.hidden_dim, self.config.rank, init);
        self.fc2 = LowRankLinear::new(self.config.hidden_dim, v, self.config.rank, init);
        self
    }

    pub fn forward(&mut self, indices: &[usize]) -> Tensor2 {
        assert_eq!(indices.len(), self.config.context);
        let emb = self.embed.forward(indices);
        let (c, d) = emb.shape();
        let mut flat = Tensor2::zeros(1, c * d);
        for i in 0..c {
            for j in 0..d {
                flat.data[[0, i * d + j]] = emb.data[[i, j]];
            }
        }
        let h = self.fc1.forward(&flat);
        let h = h.relu();
        self.fc2.forward(&h)
    }

    pub fn predict(&mut self, indices: &[usize]) -> Vec<f32> {
        let logits = self.forward(indices);
        logits.softmax().data.row(0).to_owned().to_vec()
    }

    pub fn zero_grads(&mut self) {
        self.embed.zero_grads();
        self.fc1.zero_grads();
        self.fc2.zero_grads();
    }

    pub fn apply_sgd(&mut self, lr: f32) {
        self.embed.apply_grad(lr);
        self.fc1.apply_grad(lr);
        self.fc2.apply_grad(lr);
    }

    /// Dense logical parameter count (what a dense model of this shape would have).
    pub fn logical_params(&self) -> usize {
        let e = self.config.vocab_size * self.config.embed_dim;
        e + self.fc1.logical_params() + self.fc2.logical_params()
    }

    /// Stored parameter count (the compact representation actually kept).
    pub fn stored_params(&self) -> usize {
        let e = self.config.vocab_size * self.config.embed_dim;
        e + self.fc1.stored_params() + self.fc2.stored_params()
    }

    /// Estimated size on disk after int8 quantization (bytes + header overhead).
    pub fn quantized_bytes(&self) -> usize {
        // 8-byte magic + 5 u32 config fields + 7 tensors, each with a
        // 4-byte scale and 4-byte length, then 1 byte per stored weight.
        self.stored_params() + 8 + 5 * 4 + 7 * 8
    }

    /// Save to a compact binary file with int8-quantized weights.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(MAGIC)?;
        for v in [
            self.config.vocab_size,
            self.config.embed_dim,
            self.config.hidden_dim,
            self.config.context,
            self.config.rank,
        ] {
            f.write_all(&(v as u32).to_le_bytes())?;
        }
        // Quantize each tensor: embed.weight, fc1.u, fc1.v, fc1.bias, fc2.u, fc2.v, fc2.bias
        write_quantized(&mut f, &self.embed.weight.to_flat())?;
        write_quantized(&mut f, &self.fc1.u.to_flat())?;
        write_quantized(&mut f, &self.fc1.v.to_flat())?;
        write_quantized(&mut f, &self.fc1.bias.data.to_vec())?;
        write_quantized(&mut f, &self.fc2.u.to_flat())?;
        write_quantized(&mut f, &self.fc2.v.to_flat())?;
        write_quantized(&mut f, &self.fc2.bias.data.to_vec())?;
        Ok(())
    }

    /// Load from the compact binary format produced by `save`.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a Turbo model file",
            ));
        }
        let mut read_u32 = || -> std::io::Result<usize> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b) as usize)
        };
        let vocab_size = read_u32()?;
        let embed_dim = read_u32()?;
        let hidden_dim = read_u32()?;
        let context = read_u32()?;
        let rank = read_u32()?;

        let config = TurboConfig {
            vocab_size,
            embed_dim,
            hidden_dim,
            context,
            rank,
        };
        let in_features = embed_dim * context;

        let embed_w = read_quantized(&mut f)?;
        let fc1_u = read_quantized(&mut f)?;
        let fc1_v = read_quantized(&mut f)?;
        let fc1_b = read_quantized(&mut f)?;
        let fc2_u = read_quantized(&mut f)?;
        let fc2_v = read_quantized(&mut f)?;
        let fc2_b = read_quantized(&mut f)?;

        let mut model = TurboLM::new(config);

        // Overwrite weights with dequantized values.
        fill_tensor2(&mut model.embed.weight, &embed_w);
        fill_tensor2(&mut model.fc1.u, &fc1_u);
        fill_tensor2(&mut model.fc1.v, &fc1_v);
        fill_tensor1(&mut model.fc1.bias, &fc1_b);
        fill_tensor2(&mut model.fc2.u, &fc2_u);
        fill_tensor2(&mut model.fc2.v, &fc2_v);
        fill_tensor1(&mut model.fc2.bias, &fc2_b);

        let _ = in_features;
        Ok(model)
    }
}

/// Quantize a f32 slice to int8 and write: scale (f32), len (u32), bytes.
fn write_quantized(f: &mut File, data: &[f32]) -> std::io::Result<()> {
    let max = data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
    f.write_all(&scale.to_le_bytes())?;
    f.write_all(&(data.len() as u32).to_le_bytes())?;
    let mut bytes = Vec::with_capacity(data.len());
    for &x in data {
        let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
        bytes.push(q as u8);
    }
    f.write_all(&bytes)?;
    Ok(())
}

/// Read a quantized tensor and dequantize to f32.
fn read_quantized(f: &mut File) -> std::io::Result<Vec<f32>> {
    let mut sb = [0u8; 4];
    f.read_exact(&mut sb)?;
    let scale = f32::from_le_bytes(sb);
    let mut lb = [0u8; 4];
    f.read_exact(&mut lb)?;
    let len = u32::from_le_bytes(lb) as usize;
    let mut bytes = vec![0u8; len];
    f.read_exact(&mut bytes)?;
    Ok(bytes
        .iter()
        .map(|&b| (b as i8) as f32 * scale)
        .collect())
}

fn fill_tensor2(t: &mut Tensor2, data: &[f32]) {
    let (r, c) = t.shape();
    let mut i = 0;
    for ri in 0..r {
        for ci in 0..c {
            if i < data.len() {
                t.data[[ri, ci]] = data[i];
                i += 1;
            }
        }
    }
}

fn fill_tensor1(t: &mut Tensor1, data: &[f32]) {
    for (i, v) in t.data.iter_mut().enumerate() {
        if i < data.len() {
            *v = data[i];
        }
    }
}

fn transpose(t: &Tensor2) -> Tensor2 {
    let (r, c) = t.shape();
    let mut data = ndarray::Array2::zeros((c, r));
    for i in 0..r {
        for j in 0..c {
            data[[j, i]] = t.data[[i, j]];
        }
    }
    Tensor2 { data }
}

fn bias_broadcast(bias: &Tensor1, rows: usize) -> Tensor2 {
    let mut data = ndarray::Array2::zeros((rows, bias.data.len()));
    for i in 0..rows {
        for j in 0..bias.data.len() {
            data[[i, j]] = bias.data[j];
        }
    }
    Tensor2 { data }
}

/// Trainer for TurboLM.
pub struct TurboTrainer {
    pub model: TurboLM,
    pub lr: f32,
}

impl TurboTrainer {
    pub fn new(model: TurboLM, lr: f32) -> Self {
        Self { model, lr }
    }

    /// Train on a mini-batch. All samples are processed as one matrix batch,
    /// which is much faster than a per-sample loop for the 100M-entry shape.
    pub fn train_step(&mut self, batches: &[(Vec<usize>, usize)]) -> f32 {
        if batches.is_empty() {
            return 0.0;
        }
        self.model.zero_grads();

        let b = batches.len();
        let c = self.model.config.context;
        let d = self.model.config.embed_dim;
        let mut all_indices = Vec::with_capacity(b * c);
        let mut targets = Vec::with_capacity(b);
        for (indices, target) in batches {
            assert_eq!(indices.len(), c, "context length must match model config");
            all_indices.extend_from_slice(indices);
            targets.push(*target);
        }

        // Embedding for B*context characters -> (B*context, embed_dim).
        let emb = self.model.embed.forward(&all_indices);

        // Flatten each sample's context into one row: (B, context*embed_dim).
        let mut flat = Tensor2::zeros(b, c * d);
        for bi in 0..b {
            for ci in 0..c {
                for di in 0..d {
                    flat.data[[bi, ci * d + di]] = emb.data[[bi * c + ci, di]];
                }
            }
        }

        // Forward: low-rank fc1 -> ReLU -> low-rank fc2.
        let h_pre = self.model.fc1.forward(&flat);
        let h = h_pre.relu();
        let logits = self.model.fc2.forward(&h);

        let loss = logits.cross_entropy(&targets);
        let grad_logits = logits.softmax_cross_entropy_grad(&targets);

        // Backward through fc2, ReLU, fc1, then the embedding table.
        let grad_h = self.model.fc2.backward(&grad_logits);
        let grad_h_pre = grad_h.mul(&h_pre.relu_deriv());
        let grad_flat = self.model.fc1.backward(&grad_h_pre);

        let mut grad_emb = Tensor2::zeros(b * c, d);
        for bi in 0..b {
            for ci in 0..c {
                for di in 0..d {
                    grad_emb.data[[bi * c + ci, di]] = grad_flat.data[[bi, ci * d + di]];
                }
            }
        }
        self.model.embed.backward(&grad_emb);

        // Gradients are already averaged over the batch by the cross-entropy
        // gradient, so apply the learning rate directly.
        self.model.apply_sgd(self.lr);
        loss
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        self.model.save(path)
    }
}

/// Generate text from a seed, padding the left context with spaces.
///
/// This is the shared generation path used by the CLI and the HTTP server.
pub fn generate_text(
    model: &mut TurboLM,
    tokenizer: &crate::model::Tokenizer,
    seed: &str,
    length: usize,
    temperature: f32,
) -> String {
    let context = model.config.context;
    if context == 0 {
        return seed.to_string();
    }

    let mut ctx = tokenizer.encode(seed);
    if ctx.len() > context {
        ctx = ctx[ctx.len() - context..].to_vec();
    }
    let pad = tokenizer.char_to_id.get(&' ').copied().unwrap_or(0);
    while ctx.len() < context {
        ctx.insert(0, pad);
    }

    let mut generated = seed.to_string();
    for _ in 0..length {
        let probs = model.predict(&ctx);
        let next = sample(&probs, temperature);
        generated.push(tokenizer.id_to_char.get(&next).copied().unwrap_or('?'));
        ctx.remove(0);
        ctx.push(next);
    }
    generated
}

/// Free function: sample a character index from probabilities.
pub fn sample(probs: &[f32], temperature: f32) -> usize {
    let t = temperature.max(0.01);
    let mut scaled: Vec<f32> = probs
        .iter()
        .map(|&p| if p <= 0.0 { f32::NEG_INFINITY } else { (p.ln() / t).exp() })
        .collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowrank_forward_shape() {
        let mut l = LowRankLinear::new(8, 5, 2, Init::Xavier);
        let x = Tensor2::random(3, 8, 1.0);
        let y = l.forward(&x);
        assert_eq!(y.shape(), (3, 5));
    }

    #[test]
    fn turbo_param_counts() {
        let cfg = TurboConfig {
            vocab_size: 65,
            embed_dim: 128,
            hidden_dim: 12288,
            context: 64,
            rank: 16,
        };
        let m = TurboLM::new(cfg);
        // ~100M logical params, far fewer stored.
        assert!(m.logical_params() > 90_000_000);
        assert!(m.stored_params() < 1_000_000);
        assert!(m.quantized_bytes() < 1_000_000);
    }

    #[test]
    fn hundred_million_shape_under_one_megabyte() {
        let cfg = TurboConfig {
            vocab_size: 62,
            embed_dim: 128,
            hidden_dim: 64,
            context: 64,
            rank: 30,
        };
        let m = TurboLM::new(cfg).with_logical_target(100_000_000);
        assert!(m.logical_params() >= 99_000_000);
        assert!(m.quantized_bytes() < 1_000_000);
    }

    #[test]
    fn batched_train_step_is_finite() {
        let cfg = TurboConfig {
            vocab_size: 8,
            embed_dim: 4,
            hidden_dim: 8,
            context: 3,
            rank: 2,
        };
        let mut trainer = TurboTrainer::new(TurboLM::new(cfg), 0.05);
        let batch = vec![
            (vec![0, 1, 2], 3),
            (vec![1, 2, 3], 4),
            (vec![2, 3, 4], 5),
        ];
        let loss = trainer.train_step(&batch);
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn batched_train_step_learns_tiny_pattern() {
        let cfg = TurboConfig {
            vocab_size: 4,
            embed_dim: 2,
            hidden_dim: 8,
            context: 3,
            rank: 2,
        };
        let mut trainer = TurboTrainer::new(TurboLM::new(cfg), 0.1);
        let batch = vec![
            (vec![0, 1, 2], 3),
            (vec![1, 2, 3], 0),
            (vec![2, 3, 0], 1),
            (vec![3, 0, 1], 2),
        ];
        let first = trainer.train_step(&batch);
        for _ in 0..300 {
            trainer.train_step(&batch);
        }
        let last = trainer.train_step(&batch);
        assert!(
            last < first,
            "loss should decrease on a deterministic pattern: {first} -> {last}"
        );
    }

    #[test]
    fn save_load_roundtrip() {
        let cfg = TurboConfig {
            vocab_size: 10,
            embed_dim: 4,
            hidden_dim: 8,
            context: 3,
            rank: 2,
        };
        let m = TurboLM::new(cfg);
        let path = std::env::temp_dir().join("turbo_test.bin");
        let p = path.to_str().unwrap();
        m.save(p).unwrap();
        let actual = std::fs::metadata(p).unwrap().len() as usize;
        assert_eq!(actual, m.quantized_bytes());
        let loaded = TurboLM::load(p).unwrap();
        assert_eq!(loaded.config.hidden_dim, 8);
    }
}
