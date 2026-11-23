//! Optimizers: SGD and Adam configuration holders.
//!
//! The `CharLM` applies updates directly via `apply_sgd`. These structs keep
//! hyperparameters organized; the math lives in `layers::{Linear,Embedding}`.

#[derive(Debug, Clone)]
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr, momentum: 0.0 }
    }
    pub fn with_momentum(lr: f32, momentum: f32) -> Self {
        Self { lr, momentum }
    }
}

impl Default for SGD {
    fn default() -> Self {
        Self::new(0.01)
    }
}

#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001)
    }
}
