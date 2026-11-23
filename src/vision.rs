//! Vision: a from-scratch convolutional neural network for image classification.
//!
//! No pre-trained weights. The CNN starts from random initialization
//! (Xavier/He) and learns from labeled images you provide.
//!
//! Architecture:
//!   Conv2d(in_ch -> 8, 3x3) -> ReLU -> MaxPool2d(2)
//!   Conv2d(8 -> 16, 3x3)    -> ReLU -> MaxPool2d(2)
//!   Flatten -> Linear(features -> num_classes) -> Softmax
//!
//! Backpropagation is implemented by hand through every layer.

use crate::layers::{Init, Linear};
use crate::tensor::Tensor2;
use anyhow::{Context, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 4-D tensor in NCHW layout: (batch, channels, height, width).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor4 {
    pub data: Vec<f32>,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Tensor4 {
    pub fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self {
            data: vec![0.0; n * c * h * w],
            n,
            c,
            h,
            w,
        }
    }

    #[inline]
    pub fn idx(&self, n: usize, c: usize, h: usize, w: usize) -> usize {
        ((n * self.c + c) * self.h + h) * self.w + w
    }

    #[inline]
    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        self.data[self.idx(n, c, h, w)]
    }

    #[inline]
    pub fn set(&mut self, n: usize, c: usize, h: usize, w: usize, v: f32) {
        let i = self.idx(n, c, h, w);
        self.data[i] = v;
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.n, self.c, self.h, self.w)
    }
}

/// 2-D convolution: valid padding, stride 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2d {
    pub out_channels: usize,
    pub in_channels: usize,
    pub kh: usize,
    pub kw: usize,
    pub weight: Vec<f32>, // (out_c, in_c, kh, kw)
    pub bias: Vec<f32>,   // (out_c)
    pub d_weight: Vec<f32>,
    pub d_bias: Vec<f32>,
    #[serde(skip)]
    last_input: Option<Tensor4>,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kh: usize,
        kw: usize,
        init: Init,
    ) -> Self {
        let fan_in = in_channels * kh * kw;
        let fan_out = out_channels * kh * kw;
        let scale = init.scale(fan_in, fan_out);
        let mut rng = rand::thread_rng();
        let weight = (0..out_channels * in_channels * kh * kw)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self {
            out_channels,
            in_channels,
            kh,
            kw,
            weight,
            bias: vec![0.0; out_channels],
            d_weight: vec![0.0; out_channels * in_channels * kh * kw],
            d_bias: vec![0.0; out_channels],
            last_input: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor4) -> Tensor4 {
        let (n, _, h, w) = input.shape();
        let oh = h - self.kh + 1;
        let ow = w - self.kw + 1;
        let mut out = Tensor4::zeros(n, self.out_channels, oh, ow);

        for ni in 0..n {
            for oc in 0..self.out_channels {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut sum = self.bias[oc];
                        for ic in 0..self.in_channels {
                            for ki in 0..self.kh {
                                for kj in 0..self.kw {
                                    let w_idx = ((oc * self.in_channels + ic) * self.kh + ki)
                                        * self.kw
                                        + kj;
                                    sum += input.get(ni, ic, i + ki, j + kj) * self.weight[w_idx];
                                }
                            }
                        }
                        out.set(ni, oc, i, j, sum);
                    }
                }
            }
        }

        self.last_input = Some(input.clone());
        out
    }

    pub fn backward(&mut self, grad_output: &Tensor4) -> Tensor4 {
        let input = self
            .last_input
            .as_ref()
            .expect("forward must be called before backward")
            .clone();
        let (n, _, h, w) = input.shape();
        let oh = grad_output.h;
        let ow = grad_output.w;

        let mut grad_input = Tensor4::zeros(n, input.c, h, w);
        self.d_weight = vec![0.0; self.out_channels * self.in_channels * self.kh * self.kw];
        self.d_bias = vec![0.0; self.out_channels];

        for ni in 0..n {
            for oc in 0..self.out_channels {
                for i in 0..oh {
                    for j in 0..ow {
                        let g = grad_output.get(ni, oc, i, j);
                        self.d_bias[oc] += g;
                        for ic in 0..self.in_channels {
                            for ki in 0..self.kh {
                                for kj in 0..self.kw {
                                    let w_idx = ((oc * self.in_channels + ic) * self.kh + ki)
                                        * self.kw
                                        + kj;
                                    // dW += input * g
                                    self.d_weight[w_idx] +=
                                        input.get(ni, ic, i + ki, j + kj) * g;
                                    // dInput += W * g
                                    let cur = grad_input.get(ni, ic, i + ki, j + kj);
                                    grad_input.set(
                                        ni,
                                        ic,
                                        i + ki,
                                        j + kj,
                                        cur + self.weight[w_idx] * g,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        grad_input
    }

    pub fn apply_grad(&mut self, lr: f32) {
        for (w, g) in self.weight.iter_mut().zip(self.d_weight.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.bias.iter_mut().zip(self.d_bias.iter()) {
            *b -= lr * g;
        }
    }

    pub fn zero_grads(&mut self) {
        self.d_weight.iter_mut().for_each(|x| *x = 0.0);
        self.d_bias.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// 2x2 (configurable) max pooling with stride = size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2d {
    pub size: usize,
    #[serde(skip)]
    last_input: Option<Tensor4>,
    #[serde(skip)]
    last_argmax: Option<Vec<usize>>,
}

impl MaxPool2d {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            last_input: None,
            last_argmax: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor4) -> Tensor4 {
        let oh = input.h / self.size;
        let ow = input.w / self.size;
        let mut out = Tensor4::zeros(input.n, input.c, oh, ow);
        let mut argmax = vec![0usize; input.n * input.c * oh * ow];

        for ni in 0..input.n {
            for ci in 0..input.c {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;
                        for di in 0..self.size {
                            for dj in 0..self.size {
                                let hi = i * self.size + di;
                                let wi = j * self.size + dj;
                                let v = input.get(ni, ci, hi, wi);
                                if v > max_val {
                                    max_val = v;
                                    max_idx = input.idx(ni, ci, hi, wi);
                                }
                            }
                        }
                        out.set(ni, ci, i, j, max_val);
                        let oidx = ((ni * input.c + ci) * oh + i) * ow + j;
                        argmax[oidx] = max_idx;
                    }
                }
            }
        }

        self.last_input = Some(input.clone());
        self.last_argmax = Some(argmax);
        out
    }

    pub fn backward(&mut self, grad_output: &Tensor4) -> Tensor4 {
        let input = self
            .last_input
            .as_ref()
            .expect("forward must be called before backward");
        let argmax = self
            .last_argmax
            .as_ref()
            .expect("forward must be called before backward");
        let mut grad_input = Tensor4::zeros(input.n, input.c, input.h, input.w);
        let oh = grad_output.h;
        let ow = grad_output.w;

        for ni in 0..grad_output.n {
            for ci in 0..grad_output.c {
                for i in 0..oh {
                    for j in 0..ow {
                        let oidx = ((ni * grad_output.c + ci) * oh + i) * ow + j;
                        let idx = argmax[oidx];
                        grad_input.data[idx] += grad_output.get(ni, ci, i, j);
                    }
                }
            }
        }

        grad_input
    }

    pub fn zero_grads(&mut self) {}
}

/// Elementwise ReLU on a 4-D tensor.
fn relu4(t: &Tensor4) -> Tensor4 {
    Tensor4 {
        data: t.data.iter().map(|&x| x.max(0.0)).collect(),
        ..t.clone()
    }
}

/// Backward through ReLU given the pre-activation input.
fn relu_backward4(grad: &Tensor4, pre_activation: &Tensor4) -> Tensor4 {
    Tensor4 {
        data: grad
            .data
            .iter()
            .zip(pre_activation.data.iter())
            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
            .collect(),
        ..grad.clone()
    }
}

/// Flatten a 4-D tensor (n, c, h, w) into a 2-D tensor (n, c*h*w).
fn flatten4(t: &Tensor4) -> Tensor2 {
    let features = t.c * t.h * t.w;
    let mut out = Tensor2::zeros(t.n, features);
    for ni in 0..t.n {
        for c in 0..t.c {
            for h in 0..t.h {
                for w in 0..t.w {
                    out.data[[ni, (c * t.h + h) * t.w + w]] = t.get(ni, c, h, w);
                }
            }
        }
    }
    out
}

/// Unflatten a 2-D gradient back into a 4-D tensor with the given shape.
fn unflatten2(grad: &Tensor2, shape: &Tensor4) -> Tensor4 {
    let mut out = Tensor4::zeros(shape.n, shape.c, shape.h, shape.w);
    for ni in 0..shape.n {
        for c in 0..shape.c {
            for h in 0..shape.h {
                for w in 0..shape.w {
                    out.set(ni, c, h, w, grad.data[[ni, (c * shape.h + h) * shape.w + w]]);
                }
            }
        }
    }
    out
}

/// Cached activations for backprop.
#[derive(Debug, Clone)]
pub struct VisionCache {
    pub conv1_out: Tensor4,
    pub pool1_out: Tensor4,
    pub conv2_out: Tensor4,
    pub pool2_out: Tensor4,
    pub flat: Tensor2,
}

/// The from-scratch vision CNN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCNN {
    pub conv1: Conv2d,
    pub pool1: MaxPool2d,
    pub conv2: Conv2d,
    pub pool2: MaxPool2d,
    pub fc: Linear,
    pub input_channels: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub num_classes: usize,
    #[serde(skip)]
    cache: Option<VisionCache>,
}

impl VisionCNN {
    pub fn new(
        input_channels: usize,
        input_h: usize,
        input_w: usize,
        num_classes: usize,
    ) -> Self {
        let init = Init::He;
        let conv1 = Conv2d::new(input_channels, 8, 3, 3, init);
        let pool1 = MaxPool2d::new(2);
        let conv2 = Conv2d::new(8, 16, 3, 3, init);
        let pool2 = MaxPool2d::new(2);

        // Compute feature size after conv/pool stack.
        let h1 = input_h - 3 + 1;
        let w1 = input_w - 3 + 1;
        let h2 = h1 / 2;
        let w2 = w1 / 2;
        let h3 = h2 - 3 + 1;
        let w3 = w2 - 3 + 1;
        let h4 = h3 / 2;
        let w4 = w3 / 2;
        let features = 16 * h4 * w4;

        let fc = Linear::new(features, num_classes, Init::Xavier);

        Self {
            conv1,
            pool1,
            conv2,
            pool2,
            fc,
            input_channels,
            input_h,
            input_w,
            num_classes,
            cache: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor4) -> Tensor2 {
        let conv1_out = self.conv1.forward(input);
        let relu1 = relu4(&conv1_out);
        let pool1_out = self.pool1.forward(&relu1);

        let conv2_out = self.conv2.forward(&pool1_out);
        let relu2 = relu4(&conv2_out);
        let pool2_out = self.pool2.forward(&relu2);

        let flat = flatten4(&pool2_out);
        let logits = self.fc.forward(&flat);

        self.cache = Some(VisionCache {
            conv1_out,
            pool1_out,
            conv2_out,
            pool2_out,
            flat,
        });

        logits
    }

    pub fn backward(&mut self, grad_logits: &Tensor2) {
        let cache = self
            .cache
            .as_ref()
            .expect("forward must be called before backward")
            .clone();

        // FC backward
        let grad_flat = self.fc.backward(grad_logits);

        // Unflatten to pool2 output shape
        let grad_pool2 = unflatten2(&grad_flat, &cache.pool2_out);

        // Pool2 backward
        let grad_relu2 = self.pool2.backward(&grad_pool2);

        // ReLU2 backward
        let grad_conv2 = relu_backward4(&grad_relu2, &cache.conv2_out);

        // Conv2 backward
        let grad_pool1 = self.conv2.backward(&grad_conv2);

        // Pool1 backward
        let grad_relu1 = self.pool1.backward(&grad_pool1);

        // ReLU1 backward
        let grad_conv1 = relu_backward4(&grad_relu1, &cache.conv1_out);

        // Conv1 backward
        let _ = self.conv1.backward(&grad_conv1);
    }

    pub fn apply_grad(&mut self, lr: f32) {
        self.conv1.apply_grad(lr);
        self.conv2.apply_grad(lr);
        self.fc.apply_grad(lr);
    }

    pub fn zero_grads(&mut self) {
        self.conv1.zero_grads();
        self.conv2.zero_grads();
        self.fc.zero_grads();
    }

    pub fn predict(&mut self, image: &Tensor4) -> Vec<f32> {
        let logits = self.forward(image);
        logits.softmax().data.row(0).to_owned().to_vec()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// Trainer for the vision CNN.
pub struct VisionTrainer {
    pub model: VisionCNN,
    pub lr: f32,
}

impl VisionTrainer {
    pub fn new(model: VisionCNN, lr: f32) -> Self {
        Self { model, lr }
    }

    /// Train on a batch of images (stacked) and labels.
    pub fn train_step(&mut self, images: &[Tensor4], labels: &[usize]) -> f32 {
        // Stack images into a single batch tensor.
        let n = images.len();
        let (_, c, h, w) = images[0].shape();
        let mut batch = Tensor4::zeros(n, c, h, w);
        for (ni, img) in images.iter().enumerate() {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        batch.set(ni, ci, hi, wi, img.get(0, ci, hi, wi));
                    }
                }
            }
        }

        let logits = self.model.forward(&batch);
        let loss = logits.cross_entropy(labels);
        let grad = logits.softmax_cross_entropy_grad(labels);

        self.model.zero_grads();
        self.model.backward(&grad);
        self.model.apply_grad(self.lr);

        loss
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        self.model.save(path)
    }
}

/// Load an image, resize it, and convert it to a normalized Tensor4.
pub fn load_image(
    path: &str,
    target_h: usize,
    target_w: usize,
    channels: usize,
) -> Result<Tensor4> {
    let img = image::open(Path::new(path))
        .with_context(|| format!("open image {path}"))?
        .to_rgb8();

    let resized = image::imageops::resize(
        &img,
        target_w as u32,
        target_h as u32,
        image::imageops::FilterType::Triangle,
    );

    let mut t = Tensor4::zeros(1, channels, target_h, target_w);
    for y in 0..target_h {
        for x in 0..target_w {
            let p = resized.get_pixel(x as u32, y as u32);
            if channels == 1 {
                let gray = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
                    / 255.0;
                t.set(0, 0, y, x, gray);
            } else {
                t.set(0, 0, y, x, p[0] as f32 / 255.0);
                t.set(0, 1, y, x, p[1] as f32 / 255.0);
                t.set(0, 2, y, x, p[2] as f32 / 255.0);
            }
        }
    }
    Ok(t)
}

/// Load a manifest file: one `image_path label` pair per line.
/// Blank lines and lines starting with `#` are ignored.
pub fn load_manifest(path: &str) -> Result<Vec<(String, usize)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read manifest {path}"))?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // rsplitn handles paths containing spaces; label is the last token.
        let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
        if parts.len() != 2 {
            anyhow::bail!("manifest line {} is malformed: {line}", lineno + 1);
        }
        let label: usize = parts[0]
            .parse()
            .with_context(|| format!("manifest line {}: invalid label '{}'", lineno + 1, parts[0]))?;
        let image_path = parts[1].to_string();
        out.push((image_path, label));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor4_indexing() {
        let mut t = Tensor4::zeros(1, 2, 3, 4);
        t.set(0, 1, 2, 3, 9.0);
        assert_eq!(t.get(0, 1, 2, 3), 9.0);
    }

    #[test]
    fn conv_forward_shape() {
        let mut conv = Conv2d::new(1, 4, 3, 3, Init::He);
        let input = Tensor4::zeros(2, 1, 8, 8);
        let out = conv.forward(&input);
        assert_eq!(out.shape(), (2, 4, 6, 6));
    }

    #[test]
    fn cnn_forward_shape() {
        let mut cnn = VisionCNN::new(1, 16, 16, 10);
        let input = Tensor4::zeros(1, 1, 16, 16);
        let logits = cnn.forward(&input);
        assert_eq!(logits.shape(), (1, 10));
    }

    #[test]
    fn train_step_runs() {
        let cnn = VisionCNN::new(1, 16, 16, 3);
        let mut trainer = VisionTrainer::new(cnn, 0.01);
        let img = Tensor4::zeros(1, 1, 16, 16);
        let loss = trainer.train_step(&[img], &[0]);
        assert!(loss.is_finite());
    }
}
