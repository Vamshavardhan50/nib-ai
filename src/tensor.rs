//! Minimal tensor abstraction on top of `ndarray`.
//!
//! Everything here is built from scratch — no pre-trained weights, no external
//! model files. Tensors are just f32 arrays with shape metadata.

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A 2-D tensor (matrix). The workhorse of the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor2 {
    pub data: Array2<f32>,
}

impl Tensor2 {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: Array2::zeros((rows, cols)),
        }
    }

    pub fn random(rows: usize, cols: usize, scale: f32) -> Self {
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((rows, cols), |_| {
            rng.gen_range(-scale..scale) as f32
        });
        Self { data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn from_vec(vec: Vec<Vec<f32>>) -> Self {
        let rows = vec.len();
        let cols = vec.first().map(|v| v.len()).unwrap_or(0);
        let mut data = Array2::zeros((rows, cols));
        for (i, row) in vec.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                data[[i, j]] = v;
            }
        }
        Self { data }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    pub fn rows(&self) -> usize {
        self.data.nrows()
    }

    pub fn cols(&self) -> usize {
        self.data.ncols()
    }

    /// Element-wise add.
    pub fn add(&self, other: &Tensor2) -> Tensor2 {
        Tensor2 {
            data: &self.data + &other.data,
        }
    }

    /// Element-wise multiply.
    pub fn mul(&self, other: &Tensor2) -> Tensor2 {
        Tensor2 {
            data: &self.data * &other.data,
        }
    }

    /// Matrix multiply: self @ other.
    pub fn matmul(&self, other: &Tensor2) -> Tensor2 {
        Tensor2 {
            data: self.data.dot(&other.data),
        }
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f32) -> Tensor2 {
        Tensor2 {
            data: self.data.mapv(|x| x * s),
        }
    }

    /// In-place add scaled other: self += s * other.
    pub fn add_scaled(&mut self, other: &Tensor2, s: f32) {
        self.data += &other.data.mapv(|x| x * s);
    }

    /// Sum across rows (axis 0), producing a 1-D row vector.
    pub fn sum_rows(&self) -> Tensor1 {
        Tensor1 {
            data: self.data.sum_axis(Axis(0)),
        }
    }

    /// Relu activation (in-place friendly).
    pub fn relu(&self) -> Tensor2 {
        Tensor2 {
            data: self.data.mapv(|x| x.max(0.0)),
        }
    }

    /// Relu derivative mask (1 where x > 0).
    pub fn relu_deriv(&self) -> Tensor2 {
        Tensor2 {
            data: self.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
        }
    }

    /// Tanh activation.
    pub fn tanh(&self) -> Tensor2 {
        Tensor2 {
            data: self.data.mapv(|x| x.tanh()),
        }
    }

    /// Tanh derivative: 1 - tanh^2.
    pub fn tanh_deriv(&self) -> Tensor2 {
        let t = self.tanh();
        Tensor2 {
            data: t.data.mapv(|x| 1.0 - x * x),
        }
    }

    /// Softmax across rows (each row sums to 1).
    pub fn softmax(&self) -> Tensor2 {
        let mut out = Array2::zeros(self.data.dim());
        for i in 0..self.data.nrows() {
            let row = self.data.row(i);
            let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Array1<f32> = row.mapv(|x| (x - max).exp());
            let sum: f32 = exps.sum();
            out.row_mut(i).assign(&(exps / sum));
        }
        Tensor2 { data: out }
    }

    /// Cross-entropy loss against target class indices.
    pub fn cross_entropy(&self, targets: &[usize]) -> f32 {
        let probs = self.softmax();
        let mut loss = 0.0;
        for (i, &t) in targets.iter().enumerate() {
            if t < probs.cols() {
                loss -= probs.data[[i, t]].ln().max(-50.0);
            }
        }
        loss / targets.len() as f32
    }

    /// Softmax cross-entropy gradient w.r.t. logits.
    pub fn softmax_cross_entropy_grad(&self, targets: &[usize]) -> Tensor2 {
        let probs = self.softmax();
        let mut grad = probs.data.clone();
        for (i, &t) in targets.iter().enumerate() {
            if t < grad.ncols() {
                grad[[i, t]] -= 1.0;
            }
        }
        Tensor2 { data: grad }.scale(1.0 / targets.len() as f32)
    }

    /// Clamp values.
    pub fn clamp(&self, min: f32, max: f32) -> Tensor2 {
        Tensor2 {
            data: self.data.mapv(|x| x.clamp(min, max)),
        }
    }

    /// Get a reference to a row.
    pub fn row(&self, i: usize) -> Tensor1 {
        Tensor1 {
            data: self.data.row(i).to_owned(),
        }
    }

    pub fn to_flat(&self) -> Vec<f32> {
        self.data.iter().copied().collect()
    }
}

impl fmt::Display for Tensor2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.data.nrows() {
            for j in 0..self.data.ncols() {
                write!(f, "{:.4} ", self.data[[i, j]])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// A 1-D tensor (vector).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor1 {
    pub data: Array1<f32>,
}

impl Tensor1 {
    pub fn new(len: usize) -> Self {
        Self {
            data: Array1::zeros(len),
        }
    }

    pub fn random(len: usize, scale: f32) -> Self {
        let mut rng = thread_rng();
        let data = Array1::from_shape_fn(len, |_| rng.gen_range(-scale..scale) as f32);
        Self { data }
    }

    pub fn zeros(len: usize) -> Self {
        Self::new(len)
    }

    pub fn add(&self, other: &Tensor1) -> Tensor1 {
        Tensor1 {
            data: &self.data + &other.data,
        }
    }

    pub fn scale(&self, s: f32) -> Tensor1 {
        Tensor1 {
            data: self.data.mapv(|x| x * s),
        }
    }

    pub fn add_scaled(&mut self, other: &Tensor1, s: f32) {
        self.data += &other.data.mapv(|x| x * s);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}
