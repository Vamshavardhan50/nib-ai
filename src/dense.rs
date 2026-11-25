//! Dense language model with 4-bit persistence.
//!
//! The model is a normal 2-layer MLP:
//!
//!   embed(context chars) -> flatten -> Linear -> ReLU -> Linear -> logits
//!
//! The shipped configuration uses `context=64`, `embed=128`, `hidden=10902`,
//! `vocab=62`: **90,004,008 independent weights** in **45,117,066 bytes** at
//! 4 bits per weight. The full `hidden=12114` configuration has 100,009,068
//! weights and a 50,128,686-byte file.
//!
//! The model can be trained directly with [`DenseTrainer`], initialized by
//! materializing a trained low-rank Turbo model with [`from_turbo`], or pruned
//! with [`prune_hidden`] and then fine-tuned. The final file always stores the
//! full dense matrices.

use crate::layers::Init;
use crate::model::{CharLM, ModelConfig, Tokenizer};
use crate::tensor::{Tensor1, Tensor2};
use crate::turbo::{self, TurboLM};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

const MAGIC: &[u8; 8] = b"CDENSE4\0";
const VERSION: u32 = 1;

/// Dense parameter count for a configured model.
pub fn params(config: &ModelConfig) -> usize {
    config.vocab_size * config.embed_dim
        + config.context * config.embed_dim * config.hidden_dim
        + config.hidden_dim * config.vocab_size
        + config.hidden_dim
        + config.vocab_size
}

/// Exact file size of the 4-bit format, including headers and row scales.
pub fn quantized_bytes(config: &ModelConfig) -> usize {
    let header = 8 + 6 * 4;
    header
        + matrix_bytes(config.vocab_size, config.embed_dim)
        + matrix_bytes(config.context * config.embed_dim, config.hidden_dim)
        + matrix_bytes(config.hidden_dim, config.vocab_size)
        + bias_bytes(config.hidden_dim)
        + bias_bytes(config.vocab_size)
}

fn matrix_bytes(rows: usize, cols: usize) -> usize {
    8 + rows * 4 + rows * cols.div_ceil(2)
}

fn bias_bytes(len: usize) -> usize {
    4 + len * 4
}

/// Build a dense model from a trained low-rank Turbo model.
///
/// `W = U · V` is materialized into the full `(in, out)` matrix. This keeps the
/// exact same predictions (up to f32 rounding) while turning the compact
/// factors into a real 100M-entry dense artifact.
pub fn from_turbo(turbo: &TurboLM) -> CharLM {
    let config = ModelConfig {
        vocab_size: turbo.config.vocab_size,
        embed_dim: turbo.config.embed_dim,
        hidden_dim: turbo.config.hidden_dim,
        context: turbo.config.context,
        init: Init::Xavier,
    };
    let mut model = CharLM::new(config);
    model.embed.weight.data.assign(&turbo.embed.weight.data);
    model.fc1.weight.data = turbo.fc1.u.data.dot(&turbo.fc1.v.data);
    model.fc1.bias.data.assign(&turbo.fc1.bias.data);
    model.fc2.weight.data = turbo.fc2.u.data.dot(&turbo.fc2.v.data);
    model.fc2.bias.data.assign(&turbo.fc2.bias.data);
    model
}

/// Keep the most important hidden units and return a smaller dense model.
///
/// Units are ranked by the L2 norm of their outgoing `fc2` weights, which
/// measures how strongly each one affects the logits. The kept units are
/// restored to their original order, producing a direct sub-network that can
/// be fine-tuned on real text.
pub fn prune_hidden(model: &CharLM, hidden: usize) -> anyhow::Result<CharLM> {
    let old_hidden = model.config.hidden_dim;
    if hidden == 0 {
        anyhow::bail!("hidden size must be at least 1");
    }
    if hidden > old_hidden {
        anyhow::bail!("cannot grow hidden size from {old_hidden} to {hidden}");
    }
    if hidden == old_hidden {
        return Ok(model.clone());
    }

    let vocab = model.config.vocab_size;
    let in_features = model.config.context * model.config.embed_dim;
    let mut ranked: Vec<(usize, f32)> = (0..old_hidden)
        .map(|unit| {
            let score = (0..vocab)
                .map(|v| model.fc2.weight.data[[unit, v]].powi(2))
                .sum::<f32>()
                .sqrt();
            (unit, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<usize> = ranked.into_iter().take(hidden).map(|(unit, _)| unit).collect();
    keep.sort_unstable();

    let config = ModelConfig {
        hidden_dim: hidden,
        ..model.config.clone()
    };
    let mut out = CharLM::new(config);
    out.embed.weight.data.assign(&model.embed.weight.data);
    out.fc2.bias.data.assign(&model.fc2.bias.data);
    for (new_unit, &old_unit) in keep.iter().enumerate() {
        for i in 0..in_features {
            out.fc1.weight.data[[i, new_unit]] = model.fc1.weight.data[[i, old_unit]];
        }
        out.fc1.bias.data[new_unit] = model.fc1.bias.data[old_unit];
        for v in 0..vocab {
            out.fc2.weight.data[[new_unit, v]] = model.fc2.weight.data[[old_unit, v]];
        }
    }
    Ok(out)
}

/// Save a dense model as 4-bit weights with one scale per row.
pub fn save_int4(model: &CharLM, path: &str) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(MAGIC)?;
    for value in [
        VERSION,
        model.config.vocab_size as u32,
        model.config.embed_dim as u32,
        model.config.hidden_dim as u32,
        model.config.context as u32,
        0, // reserved
    ] {
        out.write_all(&value.to_le_bytes())?;
    }
    write_matrix4(&mut out, &model.embed.weight)?;
    write_matrix4(&mut out, &model.fc1.weight)?;
    write_matrix4(&mut out, &model.fc2.weight)?;
    write_bias(&mut out, &model.fc1.bias)?;
    write_bias(&mut out, &model.fc2.bias)?;
    out.flush()
}

/// Load a dense model from the 4-bit format.
pub fn load_int4(path: &str) -> std::io::Result<CharLM> {
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a dense 4-bit model file",
        ));
    }
    let mut read_u32 = || -> std::io::Result<usize> {
        let mut bytes = [0u8; 4];
        input.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes) as usize)
    };
    let version = read_u32()?;
    if version != VERSION as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported dense model version {version}"),
        ));
    }
    let vocab_size = read_u32()?;
    let embed_dim = read_u32()?;
    let hidden_dim = read_u32()?;
    let context = read_u32()?;
    let _reserved = read_u32()?;

    let config = ModelConfig {
        vocab_size,
        embed_dim,
        hidden_dim,
        context,
        init: Init::Xavier,
    };
    let mut model = CharLM::new(config);
    let embed = read_matrix4(&mut input, vocab_size, embed_dim)?;
    let fc1_w = read_matrix4(&mut input, context * embed_dim, hidden_dim)?;
    let fc2_w = read_matrix4(&mut input, hidden_dim, vocab_size)?;
    let fc1_b = read_bias(&mut input, hidden_dim)?;
    let fc2_b = read_bias(&mut input, vocab_size)?;

    model.embed.weight = embed;
    model.fc1.weight = fc1_w;
    model.fc2.weight = fc2_w;
    model.fc1.bias = fc1_b;
    model.fc2.bias = fc2_b;
    Ok(model)
}

/// Generate text with a dense model.
pub fn generate_text(
    model: &mut CharLM,
    tokenizer: &Tokenizer,
    seed: &str,
    length: usize,
    temperature: f32,
) -> String {
    let context = model.config.context;
    if context == 0 {
        return seed.to_string();
    }
    let mut ctx = turbo::context_from_seed(tokenizer, seed, context);

    let mut generated = seed.to_string();
    for _ in 0..length {
        let probs = model.predict(&ctx);
        let next = turbo::sample(&probs, temperature);
        generated.push(tokenizer.id_to_char.get(&next).copied().unwrap_or('?'));
        ctx.remove(0);
        ctx.push(next);
    }
    generated
}

/// Batched trainer for the dense model.
pub struct DenseTrainer {
    pub model: CharLM,
    pub lr: f32,
}

impl DenseTrainer {
    pub fn new(model: CharLM, lr: f32) -> Self {
        Self { model, lr }
    }

    /// Train on a mini-batch. Uses matrix batches so the large layers are
    /// updated with efficient matrix multiplies instead of per-sample loops.
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

        let emb = self.model.embed.forward(&all_indices);
        let mut flat = Tensor2::zeros(b, c * d);
        for bi in 0..b {
            for ci in 0..c {
                for di in 0..d {
                    flat.data[[bi, ci * d + di]] = emb.data[[bi * c + ci, di]];
                }
            }
        }

        let h_pre = self.model.fc1.forward(&flat);
        let h = h_pre.relu();
        let logits = self.model.fc2.forward(&h);

        let loss = logits.cross_entropy(&targets);
        let grad_logits = logits.softmax_cross_entropy_grad(&targets);

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

        self.model.apply_sgd(self.lr);
        loss
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        save_int4(&self.model, path)
    }
}

fn write_matrix4<W: Write>(out: &mut W, tensor: &Tensor2) -> std::io::Result<()> {
    let (rows, cols) = tensor.shape();
    out.write_all(&(rows as u32).to_le_bytes())?;
    out.write_all(&(cols as u32).to_le_bytes())?;
    let mut packed = vec![0u8; cols.div_ceil(2)];
    for r in 0..rows {
        let mut max = 0.0f32;
        for c in 0..cols {
            max = max.max(tensor.data[[r, c]].abs());
        }
        let scale = if max == 0.0 { 1.0 } else { max / 7.0 };
        out.write_all(&scale.to_le_bytes())?;
        packed.fill(0);
        for c in 0..cols {
            let q = (tensor.data[[r, c]] / scale).round().clamp(-7.0, 7.0) as i8;
            let nibble = (q & 0x0f) as u8;
            if c % 2 == 0 {
                packed[c / 2] |= nibble;
            } else {
                packed[c / 2] |= nibble << 4;
            }
        }
        out.write_all(&packed)?;
    }
    Ok(())
}

fn read_matrix4<R: Read>(input: &mut R, rows: usize, cols: usize) -> std::io::Result<Tensor2> {
    let mut rows_bytes = [0u8; 4];
    let mut cols_bytes = [0u8; 4];
    input.read_exact(&mut rows_bytes)?;
    input.read_exact(&mut cols_bytes)?;
    let stored_rows = u32::from_le_bytes(rows_bytes) as usize;
    let stored_cols = u32::from_le_bytes(cols_bytes) as usize;
    if stored_rows != rows || stored_cols != cols {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "dense tensor shape mismatch: expected {rows}x{cols}, got {stored_rows}x{stored_cols}"
            ),
        ));
    }

    let mut data = ndarray::Array2::zeros((rows, cols));
    let mut packed = vec![0u8; cols.div_ceil(2)];
    for r in 0..rows {
        let mut scale_bytes = [0u8; 4];
        input.read_exact(&mut scale_bytes)?;
        let scale = f32::from_le_bytes(scale_bytes);
        input.read_exact(&mut packed)?;
        for c in 0..cols {
            let byte = packed[c / 2];
            let nibble = if c % 2 == 0 { byte & 0x0f } else { byte >> 4 } as i8;
            let value = if nibble >= 8 { nibble - 16 } else { nibble };
            data[[r, c]] = value as f32 * scale;
        }
    }
    Ok(Tensor2 { data })
}

fn write_bias<W: Write>(out: &mut W, bias: &Tensor1) -> std::io::Result<()> {
    out.write_all(&(bias.data.len() as u32).to_le_bytes())?;
    for &value in bias.data.iter() {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_bias<R: Read>(input: &mut R, len: usize) -> std::io::Result<Tensor1> {
    let mut len_bytes = [0u8; 4];
    input.read_exact(&mut len_bytes)?;
    let stored_len = u32::from_le_bytes(len_bytes) as usize;
    if stored_len != len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("dense bias length mismatch: expected {len}, got {stored_len}"),
        ));
    }
    let mut data = ndarray::Array1::zeros(len);
    let mut bytes = [0u8; 4];
    for value in data.iter_mut() {
        input.read_exact(&mut bytes)?;
        *value = f32::from_le_bytes(bytes);
    }
    Ok(Tensor1 { data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 8,
            embed_dim: 4,
            hidden_dim: 16,
            context: 3,
            init: Init::Xavier,
        }
    }

    #[test]
    fn hundred_million_dense_fits_under_100mb() {
        let config = ModelConfig {
            vocab_size: 62,
            embed_dim: 128,
            hidden_dim: 12114,
            context: 64,
            init: Init::Xavier,
        };
        let params = params(&config);
        let bytes = quantized_bytes(&config);
        assert_eq!(params, 100_009_068);
        assert!(bytes > 49_000_000, "unexpectedly small: {bytes}");
        assert!(bytes < 100_000_000, "4-bit file is too large: {bytes}");
    }

    #[test]
    fn shipped_dense_size_matches_docs() {
        let config = ModelConfig {
            vocab_size: 62,
            embed_dim: 128,
            hidden_dim: 10902,
            context: 64,
            init: Init::Xavier,
        };
        assert_eq!(params(&config), 90_004_008);
        assert_eq!(quantized_bytes(&config), 45_117_066);

        let full = ModelConfig {
            hidden_dim: 12114,
            ..config
        };
        assert_eq!(params(&full), 100_009_068);
        assert_eq!(quantized_bytes(&full), 50_128_686);
    }

    #[test]
    fn prune_hidden_shrinks_model() {
        let mut model = CharLM::new(ModelConfig {
            hidden_dim: 16,
            ..tiny_config()
        });
        let mut pruned = prune_hidden(&model, 8).unwrap();
        assert_eq!(pruned.config.hidden_dim, 8);
        assert!(params(&pruned.config) < params(&model.config));
        let logits = pruned.predict(&[0usize, 1, 2]);
        assert_eq!(logits.len(), 8);
        assert!(prune_hidden(&model, 17).is_err());
    }

    #[test]
    fn int4_roundtrip_preserves_predictions() {
        let mut model = CharLM::new(tiny_config());
        let path = std::env::temp_dir().join(format!(
            "nib-dense-test-{}.bin",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        let context = vec![0usize, 1, 2];
        let before = model.predict(&context);
        save_int4(&model, &path_str).unwrap();
        let mut loaded = load_int4(&path_str).unwrap();
        let after = loaded.predict(&context);
        let _ = std::fs::remove_file(&path);

        assert_eq!(before.len(), after.len());
        let max_diff = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.05, "4-bit predictions differ by {max_diff}");
    }

    #[test]
    fn batched_dense_train_step_is_finite() {
        let mut trainer = DenseTrainer::new(CharLM::new(tiny_config()), 0.05);
        let batch = vec![(vec![0usize, 1, 2], 3usize); 4];
        let loss = trainer.train_step(&batch);
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn from_turbo_matches_low_rank_predictions() {
        use crate::turbo::TurboConfig;

        let turbo = TurboLM::new(TurboConfig {
            vocab_size: 8,
            embed_dim: 4,
            hidden_dim: 16,
            context: 3,
            rank: 2,
        });
        let mut dense = from_turbo(&turbo);
        let mut turbo = turbo;
        let context = vec![0usize, 1, 2];
        let a = turbo.predict(&context);
        let b = dense.predict(&context);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "dense expansion differs by {max_diff}");
    }
}
