use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

///Multi-head Attention block with relative position embeddings.
#[derive(Debug, Module)]
pub struct Attention<B: Backend> {
    pub num_heads: usize,
    pub scale: f32,
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
    pub use_rel_pos: bool,
    pub rel_pos_h: Option<Param<Tensor<B, 2>>>,
    pub rel_pos_w: Option<Param<Tensor<B, 2>>>,
}
impl<B: Backend> Attention<B> {
    // Args:
    // dim (int): Number of input channels.
    // num_heads (int): Number of attention heads.
    // qkv_bias (bool):  If True, add a learnable bias to query, key, value.
    // rel_pos (bool): If True, add relative positional embeddings to the attention map.
    // rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
    // input_size (tuple(int, int) or None): Input resolution for calculating the relative
    //     positional parameter size.
    pub fn new(
        dim: usize,
        num_heads: Option<usize>,
        qkv_bias: Option<bool>,
        use_rel_pos: Option<bool>,
        _rel_pos_zero_init: Option<bool>,
        input_size: Option<Size>,
        device: &B::Device,
    ) -> Self {
        let num_heads = num_heads.unwrap_or(8);
        let qkv_bias = qkv_bias.unwrap_or(true);
        let use_rel_pos = use_rel_pos.unwrap_or(false);
        let _rel_pos_zero_init = _rel_pos_zero_init.unwrap_or(true);

        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);
        let qkv = LinearConfig::new(dim, 3 * dim)
            .with_bias(qkv_bias)
            .init(device);
        let proj = LinearConfig::new(dim, dim).init(device);
        let mut rel_pos_h = None;
        let mut rel_pos_w = None;
        if use_rel_pos {
            assert!(
                input_size.is_some(),
                "Input size must be provided if using relative positional encoding."
            );
            let Size(h, w) = input_size.unwrap();
            rel_pos_h = Some(Param::from_tensor(Tensor::zeros(
                [2 * h - 1, head_dim],
                device,
            )));
            rel_pos_w = Some(Param::from_tensor(Tensor::zeros(
                [2 * w - 1, head_dim],
                device,
            )));
        }

        Self {
            num_heads,
            scale,
            qkv,
            proj,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = x.dims();
        let (b, h, w) = (shape[0], shape[1], shape[2]);

        let qkv = self
            .qkv
            .forward(x)
            .reshape_max([b, h * w, 3, self.num_heads, usize::MAX])
            .permute([2, 0, 3, 1, 4]);
        let qkv = qkv.reshape_max([3, b * self.num_heads, h * w, usize::MAX]);
        let q = qkv.clone().narrow(0, 0, 1).squeeze::<3>();
        let k = qkv.clone().narrow(0, 1, 1).squeeze::<3>();
        let v = qkv.narrow(0, 2, 1).squeeze::<3>();

        let mut attn = (q.clone() * self.scale).matmul(k.transpose());
        if self.use_rel_pos {
            attn = add_decomposed_rel_pos(
                attn,
                q,
                self.rel_pos_h.clone().unwrap().val(),
                self.rel_pos_w.clone().unwrap().val(),
                Size(h, w),
                Size(h, w),
            )
        };
        attn = softmax(attn, 2);

        let x = attn
            .matmul(v)
            .reshape_max([b, self.num_heads, h, w, usize::MAX])
            .permute([0, 2, 3, 1, 4])
            .reshape_max([b, h, w, usize::MAX]);
        let x = self.proj.forward(x);
        x
    }
}

// Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
// https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
// Args:
//     attn (Tensor): attention map.
//     q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
//     rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
//     rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
//     q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
//     k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

// Returns:
//     attn (Tensor): attention map with added relative positional embeddings.
fn add_decomposed_rel_pos<B: Backend>(
    attn: Tensor<B, 3>,
    q: Tensor<B, 3>,
    rel_pos_h: Tensor<B, 2>,
    rel_pos_w: Tensor<B, 2>,
    q_size: Size,
    k_size: Size,
) -> Tensor<B, 3> {
    let Size(q_h, q_w) = q_size;
    let Size(k_h, k_w) = k_size;
    let rh = get_rel_pos(q_h, k_h, rel_pos_h);
    let rw = get_rel_pos(q_w, k_w, rel_pos_w);

    let shape = q.dims();
    let (b, dim) = (shape[0], shape[2]);
    let r_q = q.reshape([b, q_h, q_w, dim]);

    // einsum "bhwc,hkc->bhwk": rel_h[b,h,w,k] = sum_c r_q[b,h,w,c] * Rh[h,k,c]
    // Use broadcasting: r_q [B, q_h, q_w, 1, dim] * Rh [1, q_h, 1, k_h, dim] -> sum over dim
    let r_q_expanded: Tensor<B, 5> = r_q.clone().unsqueeze_dim(3); // [B, q_h, q_w, 1, dim]
    let rh_dims = rh.dims();
    let rh_expanded: Tensor<B, 5> = rh.reshape([1, rh_dims[0], 1, rh_dims[1], rh_dims[2]]); // [1, q_h, 1, k_h, dim]
    let rel_h: Tensor<B, 4> = (r_q_expanded.clone() * rh_expanded).sum_dims_squeeze(&[4]); // [B, q_h, q_w, k_h]

    // einsum "bhwc,wkc->bhwk": rel_w[b,h,w,k] = sum_c r_q[b,h,w,c] * Rw[w,k,c]
    // Use broadcasting: r_q [B, q_h, q_w, 1, dim] * Rw [1, 1, q_w, k_w, dim] -> sum over dim
    let rw_dims = rw.dims();
    let rw_expanded: Tensor<B, 5> = rw.reshape([1, 1, rw_dims[0], rw_dims[1], rw_dims[2]]); // [1, 1, q_w, k_w, dim]
    let rel_w: Tensor<B, 4> = (r_q_expanded * rw_expanded).sum_dims_squeeze(&[4]); // [B, q_h, q_w, k_w]

    // rel_h: [b, q_h, q_w, k_h] -> reshape to [b, q_h, q_w, k_h, 1]
    // rel_w: [b, q_h, q_w, k_w] -> reshape to [b, q_h, q_w, 1, k_w]
    let rel_h_expanded: Tensor<B, 5> = rel_h.reshape([b, q_h, q_w, k_h, 1]);
    let rel_w_expanded: Tensor<B, 5> = rel_w.reshape([b, q_h, q_w, 1, k_w]);

    let attn = attn.reshape([b, q_h, q_w, k_h, k_w]) + rel_h_expanded + rel_w_expanded;
    attn.reshape([b, q_h * q_w, k_h * k_w])
}

// Get relative positional embeddings according to the relative positions of
// query and key sizes.
// Args:
// q_size (int): size of query q.
// k_size (int): size of key k.
// rel_pos (Tensor): relative position embeddings (L, C).

// Returns:
// Extracted positional embeddings according to relative positions.
fn get_rel_pos<B: Backend>(q_size: usize, k_size: usize, rel_pos: Tensor<B, 2>) -> Tensor<B, 3> {
    let device = rel_pos.device();
    let rel_pos_dims = rel_pos.dims();
    let max_rel_dist = 2 * q_size.max(k_size) - 1;

    // Interpolate rel_pos if needed (mimics F.interpolate with mode='linear', align_corners=False)
    let rel_pos_resized = if rel_pos_dims[0] != max_rel_dist {
        let old_size = rel_pos_dims[0];
        let new_size = max_rel_dist;
        let embedding_dim = rel_pos_dims[1];

        // PyTorch: rel_pos.reshape(1, L, C).permute(0, 2, 1) -> interpolate -> reshape(-1, new_size).permute(1, 0)
        // We need to interpolate each embedding dimension independently
        // rel_pos: [old_size, embedding_dim] -> [embedding_dim, old_size] for interpolation
        let rel_pos_t = rel_pos.transpose(); // [embedding_dim, old_size]

        // Create interpolation indices using tensor operations
        // For align_corners=False: pos = (i + 0.5) * scale - 0.5
        let scale = old_size as f32 / new_size as f32;
        let indices: Tensor<B, 1> = Tensor::arange(0..new_size as i64, &device)
            .float()
            .add_scalar(0.5)
            .mul_scalar(scale)
            .sub_scalar(0.5)
            .clamp_min(0.0);

        // Get floor and ceil indices
        let indices_floor = indices.clone().floor();
        let indices_ceil = indices.clone().ceil().clamp_max((old_size - 1) as f32);
        let weights = indices - indices_floor.clone();

        // Convert to integer indices for gathering
        let indices_floor_int: Tensor<B, 1, burn::tensor::Int> = indices_floor.round().int();
        let indices_ceil_int: Tensor<B, 1, burn::tensor::Int> = indices_ceil.round().int();

        // Gather values at floor and ceil indices for each embedding dimension
        // rel_pos_t: [embedding_dim, old_size]
        let values_floor = rel_pos_t.clone().select(1, indices_floor_int); // [embedding_dim, new_size]
        let values_ceil = rel_pos_t.clone().select(1, indices_ceil_int); // [embedding_dim, new_size]

        // Linear interpolation: val0 * (1 - weight) + val1 * weight
        let weights_expanded = weights.unsqueeze().repeat_dim(0, embedding_dim); // [embedding_dim, new_size]
        let ones_minus_weights = weights_expanded.clone().neg().add_scalar(1.0);
        let interpolated = values_floor * ones_minus_weights + values_ceil * weights_expanded;

        // Transpose back to [new_size, embedding_dim]
        interpolated.transpose()
    } else {
        rel_pos
    };

    // Calculate relative coordinates using tensor operations
    let q_coords: Tensor<B, 2> = Tensor::arange(0..q_size as i64, &device)
        .float()
        .reshape([q_size, 1])
        .mul_scalar((k_size as f32 / q_size as f32).max(1.0));

    let k_coords: Tensor<B, 2> = Tensor::arange(0..k_size as i64, &device)
        .float()
        .reshape([1, k_size])
        .mul_scalar((q_size as f32 / k_size as f32).max(1.0));

    let relative_coords = (q_coords.repeat_dim(1, k_size) - k_coords.repeat_dim(0, q_size))
        + (k_size as f32 - 1.) * (q_size as f32 / k_size as f32).max(1.0);

    // Use select to gather rows from rel_pos_resized
    // relative_coords: [q_size, k_size], rel_pos_resized: [max_rel_dist, embedding_dim]
    // We need to gather for each position in relative_coords
    let embedding_dim = rel_pos_resized.dims()[1];

    // Flatten relative_coords and clamp indices
    let relative_coords_flat = relative_coords
        .reshape([q_size * k_size])
        .round()
        .clamp(0.0, (max_rel_dist - 1) as f32);

    // Convert to integer indices
    let indices: Tensor<B, 1, burn::tensor::Int> = relative_coords_flat.int();

    // Gather rows from rel_pos_resized using select
    let gathered = rel_pos_resized.select(0, indices); // [q_size * k_size, embedding_dim]

    // Reshape to [q_size, k_size, embedding_dim]
    gathered.reshape([q_size, k_size, embedding_dim])
}

#[cfg(test)]
pub mod test {

    use pyo3::{types::PyAnyMethods, PyResult, Python};

    use crate::{
        python::module_to_file::module_to_file,
        python::python_data::{random_python_tensor, PythonData},
        sam_predictor::Size,
        tests::helpers::{load_module, TestBackend},
    };

    #[test]
    fn test_get_rel_pos() {
        fn python() -> PyResult<(PythonData<2>, PythonData<3>)> {
            Python::attach(|py| {
                let module = py
                    .import("segment_anything.modeling.image_encoder")?
                    .getattr("get_rel_pos")?;

                let input = random_python_tensor(py, [127, 40])?;
                let output = module.call1((32, 32, &input))?;
                Ok((input.try_into()?, output.try_into()?))
            })
        }
        let (input, python) = python().unwrap();
        let output = super::get_rel_pos::<TestBackend>(32, 32, input.into());
        python.almost_equal(output, None);
    }

    #[test]
    fn test_add_decomposed_rel_pos() {
        fn python() -> PyResult<(
            PythonData<3>,
            PythonData<3>,
            PythonData<2>,
            PythonData<2>,
            PythonData<3>,
        )> {
            Python::attach(|py| {
                let module = py
                    .import("segment_anything.modeling.image_encoder")?
                    .getattr("add_decomposed_rel_pos")?;

                let attn = random_python_tensor(py, [200, 49, 49])?;
                let q = random_python_tensor(py, [200, 49, 20])?;
                let rel_pos_h = random_python_tensor(py, [20, 20])?;
                let rel_pos_w = random_python_tensor(py, [20, 20])?;
                let output = module.call1((&attn, &q, &rel_pos_h, &rel_pos_w, (7, 7), (7, 7)))?;
                Ok((
                    attn.try_into()?,
                    q.try_into()?,
                    rel_pos_h.try_into()?,
                    rel_pos_w.try_into()?,
                    output.try_into()?,
                ))
            })
        }
        let (attn, q, rel_pos_h, rel_pos_w, python) = python().unwrap();
        let q_size = Size(7, 7);
        let k_size = Size(7, 7);
        let output = super::add_decomposed_rel_pos::<TestBackend>(
            attn.into(),
            q.into(),
            rel_pos_h.into(),
            rel_pos_w.into(),
            q_size,
            k_size,
        );
        python.almost_equal(output, None);
    }

    #[test]
    fn test_attention() {
        const FILE: &str = "attention";

        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::attach(|py| {
                let module = py
                    .import("segment_anything.modeling.image_encoder")?
                    .getattr("Attention")?;
                let module = module.call1((320, 16, true, true, true, (14, 14)))?;
                module_to_file(FILE, py, &module).unwrap();

                let input = random_python_tensor(py, [25, 14, 14, 320])?;
                let output = module.call1((&input,))?;
                Ok((input.try_into()?, output.try_into()?))
            })
        }
        let (input, python) = python().unwrap();
        let device = Default::default();
        let mut attention = super::Attention::<TestBackend>::new(
            320,
            Some(16),
            Some(true),
            Some(true),
            Some(true),
            Some(Size(14, 14)),
            &device,
        );
        attention = load_module(FILE, attention);

        // Forward
        let output = attention.forward(input.into());
        python.almost_equal(output, None);
    }
}
