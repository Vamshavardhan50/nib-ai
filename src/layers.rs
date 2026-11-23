//! Layers: Linear, Embedding.
//!
//! All weights are initialized from scratch using Xavier/Glorot or He
//! initialization — no pre-trained model files. Each layer stores its own
//! gradients after backward() so an optimizer can update them.

use crate::tensor::{Tensor1, Tensor2};
use serde::{Deserialize, Serialize};

/// Weight initialization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Init {
    /// Xavier/Glorot uniform: good for tanh/sigmoid.
    Xavier,
    /// He uniform: good for ReLU.
    He,
    /// LeCun uniform: good for selu.
    Lecun,
    /// Constant small value.
    Constant(f32),
}

impl Init {
    pub fn scale(self, fan_in: usize, fan_out: usize) -> f32 {
        match self {
            Init::Xavier => (6.0 / (fan_in + fan_out) as f32).sqrt(),
            Init::He => (6.0 / fan_in as f32).sqrt(),
            Init::Lecun => (3.0 / fan_in as f32).sqrt(),
            Init::Constant(c) => c,
        }
    }
}

/// A fully-connected (dense) layer: y = x @ W + b.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub weight: Tensor2,   // (in_features, out_features)
    pub bias: Tensor1,     // (out_features,)
    pub init: Init,
    // Gradients (filled by backward)
    pub d_weight: Tensor2,
    pub d_bias: Tensor1,
    // Cache for backprop
    last_input: Option<Tensor2>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, init: Init) -> Self {
        let scale = init.scale(in_features, out_features);
        Self {
            weight: Tensor2::random(in_features, out_features, scale),
            bias: Tensor1::zeros(out_features),
            init,
            d_weight: Tensor2::zeros(in_features, out_features),
            d_bias: Tensor1::zeros(out_features),
            last_input: None,
        }
    }

    pub fn forward(&mut self, x: &Tensor2) -> Tensor2 {
        self.last_input = Some(x.clone());
        let mut out = x.matmul(&self.weight);
        out = out.add(&self.bias_broadcast(x.rows()));
        out
    }

    /// Backprop: given grad_output (dL/dy), accumulate dL/dW, dL/db, and
    /// return dL/dx for the previous layer.
    pub fn backward(&mut self, grad_output: &Tensor2) -> Tensor2 {
        let x = self.last_input.as_ref().expect("forward called before backward");

        // dW += x^T @ grad_output  (accumulate across mini-batch)
        let dw = transpose(x).matmul(grad_output);
        self.d_weight.add_scaled(&dw, 1.0);

        // db += sum over batch of grad_output
        let db = grad_output.sum_rows();
        self.d_bias.add_scaled(&db, 1.0);

        // dx = grad_output @ W^T
        let wt = transpose(&self.weight);
        let dx = grad_output.matmul(&wt);

        dx
    }

    /// Apply an optimizer update (subtract lr * gradient).
    pub fn apply_grad(&mut self, lr: f32) {
        self.weight.add_scaled(&self.d_weight, -lr);
        self.bias.add_scaled(&self.d_bias, -lr);
    }

    fn bias_broadcast(&self, rows: usize) -> Tensor2 {
        let mut data = ndarray::Array2::zeros((rows, self.bias.data.len()));
        for i in 0..rows {
            for j in 0..self.bias.data.len() {
                data[[i, j]] = self.bias.data[j];
            }
        }
        Tensor2 { data }
    }

    pub fn zero_grads(&mut self) {
        self.d_weight.data.fill(0.0);
        self.d_bias.data.fill(0.0);
    }
}

/// Embedding layer: looks up dense vectors for integer indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub weight: Tensor2,   // (vocab_size, embed_dim)
    pub init: Init,
    pub d_weight: Tensor2,
    last_indices: Option<Vec<usize>>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize, init: Init) -> Self {
        let scale = init.scale(vocab_size, embed_dim);
        Self {
            weight: Tensor2::random(vocab_size, embed_dim, scale),
            init,
            d_weight: Tensor2::zeros(vocab_size, embed_dim),
            last_indices: None,
        }
    }

    pub fn forward(&mut self, indices: &[usize]) -> Tensor2 {
        let mut data = ndarray::Array2::zeros((indices.len(), self.weight.cols()));
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.weight.rows() {
                for j in 0..self.weight.cols() {
                    data[[i, j]] = self.weight.data[[idx, j]];
                }
            }
        }
        self.last_indices = Some(indices.to_vec());
        Tensor2 { data }
    }

    pub fn backward(&mut self, grad_output: &Tensor2) {
        // Accumulate sparse gradients into d_weight (zeroed at batch start).
        if let Some(indices) = &self.last_indices {
            for (i, &idx) in indices.iter().enumerate() {
                if idx < self.weight.rows() {
                    for j in 0..self.weight.cols() {
                        self.d_weight.data[[idx, j]] += grad_output.data[[i, j]];
                    }
                }
            }
        }
    }

    pub fn apply_grad(&mut self, lr: f32) {
        self.weight.add_scaled(&self.d_weight, -lr);
    }

    pub fn zero_grads(&mut self) {
        self.d_weight.data.fill(0.0);
    }
}

/// Transpose a 2-D tensor.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_forward_shape() {
        let mut layer = Linear::new(3, 4, Init::Xavier);
        let x = Tensor2::random(2, 3, 1.0);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), (2, 4));
    }

    #[test]
    fn embedding_lookup_shape() {
        let mut emb = Embedding::new(10, 5, Init::Xavier);
        let y = emb.forward(&[0, 1, 2]);
        assert_eq!(y.shape(), (3, 5));
    }
}
