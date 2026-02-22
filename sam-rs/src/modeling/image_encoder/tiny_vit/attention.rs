use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, TensorData},
};

/// TinyViT Attention module
///
/// Unlike standard multi-head attention, TinyViT uses:
/// - Separate dimensions for keys/queries (key_dim * num_heads) and values (attn_ratio * key_dim * num_heads)
/// - Position-based attention biases computed from relative positions
/// - LayerNorm before attention
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    norm: LayerNorm<B>,
    qkv: Linear<B>,
    proj: Linear<B>,

    // Attention parameters
    num_heads: usize,
    scale: f32,
    key_dim: usize,
    d: usize,  // value dimension per head (attn_ratio * key_dim)
    dh: usize, // total value dimension (d * num_heads)

    // Position biases
    attention_biases: Tensor<B, 2>, // [num_heads, num_attention_offsets]
    attention_bias_idxs: Tensor<B, 2, burn::tensor::Int>, // [N, N] where N = resolution[0] * resolution[1]
}

impl<B: Backend> Attention<B> {
    /// Creates a new Attention module
    ///
    /// # Arguments
    /// * `dim` - Input dimension
    /// * `key_dim` - Dimension for each attention head's key/query
    /// * `num_heads` - Number of attention heads
    /// * `attn_ratio` - Ratio of value dimension to key dimension (typically 4)
    /// * `resolution` - Spatial resolution (height, width) for computing position biases
    /// * `device` - Device for computation
    pub fn new(
        dim: usize,
        key_dim: usize,
        num_heads: usize,
        attn_ratio: usize,
        resolution: (usize, usize),
        device: &B::Device,
    ) -> Self {
        assert_eq!(
            resolution.0, resolution.1,
            "Only square resolutions supported for now"
        );

        let nh_kd = key_dim * num_heads; // Total key/query dimension
        let d = attn_ratio * key_dim; // Value dimension per head
        let dh = d * num_heads; // Total value dimension
        let h = dh + nh_kd * 2; // Total QKV dimension

        let scale = (key_dim as f32).powf(-0.5);

        // Create attention bias indices
        let (attention_biases, attention_bias_idxs) =
            Self::create_attention_biases(num_heads, resolution, device);

        Self {
            norm: LayerNormConfig::new(dim).init(device),
            qkv: LinearConfig::new(dim, h).init(device),
            proj: LinearConfig::new(dh, dim).init(device),
            num_heads,
            scale,
            key_dim,
            d,
            dh,
            attention_biases,
            attention_bias_idxs,
        }
    }

    /// Creates attention biases based on relative positions
    fn create_attention_biases(
        num_heads: usize,
        resolution: (usize, usize),
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2, burn::tensor::Int>) {
        use std::collections::HashMap;

        let (h, w) = resolution;
        let n = h * w;

        // Generate all position pairs and their relative offsets
        let mut attention_offsets = HashMap::new();
        let mut idxs = Vec::new();

        for i1 in 0..h {
            for j1 in 0..w {
                for i2 in 0..h {
                    for j2 in 0..w {
                        let offset = (
                            (i1 as i32 - i2 as i32).abs() as usize,
                            (j1 as i32 - j2 as i32).abs() as usize,
                        );
                        let len = attention_offsets.len();
                        let idx = *attention_offsets.entry(offset).or_insert(len);
                        idxs.push(idx as i64);
                    }
                }
            }
        }

        let num_offsets = attention_offsets.len();

        // Initialize attention biases to zero (will be loaded from weights)
        let attention_biases = Tensor::zeros([num_heads, num_offsets], device);

        // Create index tensor [N, N]
        let attention_bias_idxs: Tensor<B, 2, burn::tensor::Int> =
            Tensor::from_data(TensorData::new(idxs, [n, n]), device);

        (attention_biases, attention_bias_idxs)
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [B, N, C] where N = H * W
    ///
    /// # Returns
    /// * Output tensor of shape [B, N, C]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, _c] = x.dims();

        // Normalization
        let x = self.norm.forward(x);

        // Project to Q, K, V
        let qkv = self.qkv.forward(x); // [B, N, h]

        // Split into Q, K, V and reshape
        // qkv has shape [B, N, dh + nh_kd + nh_kd]
        // We need to split it into [q: key_dim, k: key_dim, v: d] per head

        let qkv = qkv.reshape([b, n, self.num_heads, self.key_dim * 2 + self.d]);

        // Split along last dimension
        let q = qkv
            .clone()
            .slice([0..b, 0..n, 0..self.num_heads, 0..self.key_dim]);
        let k = qkv.clone().slice([
            0..b,
            0..n,
            0..self.num_heads,
            self.key_dim..(self.key_dim * 2),
        ]);
        let v = qkv.slice([
            0..b,
            0..n,
            0..self.num_heads,
            (self.key_dim * 2)..(self.key_dim * 2 + self.d),
        ]);

        // Permute to [B, num_heads, N, dim_per_head]
        let q = q.swap_dims(1, 2); // [B, num_heads, N, key_dim]
        let k = k.swap_dims(1, 2); // [B, num_heads, N, key_dim]
        let v = v.swap_dims(1, 2); // [B, num_heads, N, d]

        // Compute attention scores: Q @ K^T * scale
        let attn = q.matmul(k.transpose()) * self.scale; // [B, num_heads, N, N]

        // Add attention biases
        let attn = self.add_attention_bias(attn); // [B, num_heads, N, N]

        // Apply softmax
        let attn = burn::tensor::activation::softmax(attn, 3);

        // Apply attention to values: attn @ V
        let x = attn.matmul(v); // [B, num_heads, N, d]

        // Reshape back to [B, N, dh]
        let x = x.swap_dims(1, 2); // [B, N, num_heads, d]
        let x = x.reshape([b, n, self.dh]);

        // Project output
        let x = self.proj.forward(x);

        x
    }

    /// Add position-based attention biases
    fn add_attention_bias(&self, attn: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, _num_heads, n, _n2] = attn.dims();

        // attention_biases: [num_heads, num_offsets]
        // attention_bias_idxs: [N, N]
        // We need to create: [B, num_heads, N, N]

        // Use select to gather the biases
        // For each position in [N, N], we select from the num_offsets dimension
        let indices = self.attention_bias_idxs.clone();

        // Flatten indices and gather
        let indices_flat = indices.reshape([n * n]);
        let bias_selected = self
            .attention_biases
            .clone()
            .select(1, indices_flat) // [num_heads, N*N]
            .reshape([1, self.num_heads, n, n]) // [1, num_heads, N, N]
            .repeat(&[b, 1, 1, 1]); // [B, num_heads, N, N]

        attn + bias_selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        modeling::image_encoder::tiny_vit::test_helpers::save_module_with_bn_fix,
        python::python_data::{init_torch, random_python_tensor, PythonData},
        tests::helpers::{load_module, TestBackend},
    };
    use pyo3::{types::PyAnyMethods, PyResult, Python};

    #[test]
    fn test_attention_basic() {
        const FILE: &str = "tiny_vit_attention_basic";

        fn python() -> PyResult<(PythonData<3>, PythonData<3>)> {
            Python::attach(|py| {
                // Initialize clean test environment with deterministic seed
                init_torch(py, 42)?;

                // Import TinyViT Attention from mobile_sam
                let code = r#"
import sys
sys.path.insert(0, 'mobile-sam')
from mobile_sam.modeling.tiny_vit_sam import Attention
import torch
"#;
                py.run(&std::ffi::CString::new(code).unwrap(), None, None)?;

                // Create TinyViT Attention module
                // Parameters: dim=64, key_dim=16, num_heads=4, attn_ratio=4, resolution=(7,7)
                let dim = 64;
                let key_dim = 16;
                let num_heads = 4;
                let attn_ratio = 4;
                let resolution = (7, 7);

                // Instantiate the Attention module
                let attention_cls =
                    py.eval(&std::ffi::CString::new("Attention").unwrap(), None, None)?;
                let attention =
                    attention_cls.call((dim, key_dim, num_heads, attn_ratio, resolution), None)?;

                // Save module (LayerNorm also needs weight→gamma, bias→beta renaming)
                save_module_with_bn_fix(py, FILE, &attention, &["norm"])?;

                // Create input and run forward pass
                let input = random_python_tensor(py, [2, 49, 64])?;

                let torch = py.import("torch")?;
                torch.call_method1("set_grad_enabled", (false,))?;
                attention.call_method0("eval")?;

                let output = attention.call1((input.clone(),))?;

                Ok((input.try_into()?, output.try_into()?))
            })
        }

        let (input, python_output) = python().unwrap();
        let device = Default::default();

        let mut attention = Attention::<TestBackend>::new(64, 16, 4, 4, (7, 7), &device);
        attention = load_module(FILE, attention);

        let output = attention.forward(input.into());

        python_output.almost_equal(output, None);
    }
}
