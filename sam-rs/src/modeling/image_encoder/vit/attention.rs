use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, ElementConversion, Tensor, TensorData},
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
        let q = qkv.clone().narrow(0, 0, 1).squeeze::<3>(0);
        let k = qkv.clone().narrow(0, 1, 1).squeeze::<3>(0);
        let v = qkv.narrow(0, 2, 1).squeeze::<3>(0);

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

    // Replace einsum "bhwc,hkc->bhwk" with equivalent operations
    // For each query height hi, compute: r_q[:, hi, :, :] @ rh[hi, :, :].T
    // r_q[:, hi, :, :] has shape [b, q_w, dim]
    // rh[hi, :, :] has shape [k_h, dim], transposed to [dim, k_h]
    // Result for each hi: [b, q_w, k_h]
    let mut rel_h_slices: Vec<Tensor<B, 4>> = Vec::with_capacity(q_h);
    for hi in 0..q_h {
        let r_q_slice: Tensor<B, 3> = r_q.clone().narrow(1, hi, 1).squeeze::<3>(1); // [b, q_w, dim]
        let rh_slice: Tensor<B, 2> = rh.clone().narrow(0, hi, 1).squeeze::<2>(0); // [k_h, dim]
        let rh_slice_t: Tensor<B, 2> = rh_slice.transpose(); // [dim, k_h]
                                                             // Broadcast rh_slice_t to match batch dimension: repeat for each batch
        let rh_slice_broadcast = rh_slice_t.clone().unsqueeze().repeat_dim(0, b); // [b, dim, k_h]
        let result: Tensor<B, 3> = r_q_slice.matmul(rh_slice_broadcast); // [b, q_w, k_h]
                                                                         // Reshape to [b, 1, q_w, k_h] for concatenation along dimension 1
        let result_reshaped: Tensor<B, 4> = result.reshape([b, 1, q_w, k_h]);
        rel_h_slices.push(result_reshaped);
    }
    let rel_h: Tensor<B, 4> = Tensor::cat(rel_h_slices, 1); // [b, q_h, q_w, k_h]

    // Replace einsum "bhwc,wkc->bhwk" with equivalent operations
    // For each query width wi, compute: r_q[:, :, wi, :] @ rw[wi, :, :].T
    // r_q[:, :, wi, :] has shape [b, q_h, dim]
    // rw[wi, :, :] has shape [k_w, dim], transposed to [dim, k_w]
    // Result for each wi: [b, q_h, k_w]
    let mut rel_w_slices: Vec<Tensor<B, 4>> = Vec::with_capacity(q_w);
    for wi in 0..q_w {
        let r_q_slice: Tensor<B, 3> = r_q.clone().narrow(2, wi, 1).squeeze::<3>(2); // [b, q_h, dim]
        let rw_slice: Tensor<B, 2> = rw.clone().narrow(0, wi, 1).squeeze::<2>(0); // [k_w, dim]
        let rw_slice_t: Tensor<B, 2> = rw_slice.transpose(); // [dim, k_w]
                                                             // Broadcast rw_slice_t to match batch dimension: repeat for each batch
        let rw_slice_broadcast = rw_slice_t.clone().unsqueeze().repeat_dim(0, b); // [b, dim, k_w]
        let result: Tensor<B, 3> = r_q_slice.matmul(rw_slice_broadcast); // [b, q_h, k_w]
                                                                         // Reshape to [b, q_h, 1, k_w] for concatenation along dimension 2
        let result_reshaped: Tensor<B, 4> = result.reshape([b, q_h, 1, k_w]);
        rel_w_slices.push(result_reshaped);
    }
    let rel_w: Tensor<B, 4> = Tensor::cat(rel_w_slices, 2); // [b, q_h, q_w, k_w]

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
        let rel_pos_data = rel_pos.to_data();

        // Interpolate each embedding dimension independently
        // PyTorch: rel_pos.reshape(1, L, C).permute(0, 2, 1) -> interpolate -> reshape(-1, new_size).permute(1, 0)
        let mut result = Vec::with_capacity(new_size * embedding_dim);

        for j in 0..embedding_dim {
            for i in 0..new_size {
                // F.interpolate with align_corners=False: pos = (i + 0.5) * scale - 0.5
                let scale = old_size as f32 / new_size as f32;
                let pos = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
                let idx0 = pos.floor() as usize;
                let idx1 = (idx0 + 1).min(old_size - 1);
                let weight = pos - idx0 as f32;

                let val0 =
                    rel_pos_data.as_slice::<B::FloatElem>().unwrap()[idx0 * embedding_dim + j];
                let val1 =
                    rel_pos_data.as_slice::<B::FloatElem>().unwrap()[idx1 * embedding_dim + j];
                let interp_val = val0.elem::<f32>() * (1.0 - weight) + val1.elem::<f32>() * weight;
                result.push(B::FloatElem::from_elem(interp_val));
            }
        }

        // Transpose from [embedding_dim, new_size] to [new_size, embedding_dim]
        let mut transposed = Vec::with_capacity(new_size * embedding_dim);
        for i in 0..new_size {
            for j in 0..embedding_dim {
                transposed.push(result[j * new_size + i]);
            }
        }

        Tensor::from_data(
            TensorData::new(transposed, [new_size, embedding_dim]),
            &device,
        )
    } else {
        rel_pos
    };

    // Calculate relative coordinates
    // q_coords should be [test_add_decomposed_rel_posq_size, 1], k_coords should be [1, k_size]
    let q_coords: Tensor<B, 2> = Tensor::arange(0..q_size as i64, &device)
        .float()
        .reshape([q_size, 1]) // [q_size] -> [q_size, 1]
        .mul_scalar((k_size as f32 / q_size as f32).max(1.0));

    let k_coords: Tensor<B, 2> = Tensor::arange(0..k_size as i64, &device)
        .float()
        .reshape([1, k_size]) // [k_size] -> [1, k_size]
        .mul_scalar((q_size as f32 / k_size as f32).max(1.0));

    // Manually broadcast to [q_size, k_size] before subtraction
    let q_coords_broadcast = q_coords.repeat_dim(1, k_size); // [q_size, 1] -> [q_size, k_size]
    let k_coords_broadcast = k_coords.repeat_dim(0, q_size); // [1, k_size] -> [q_size, k_size]

    let relative_coords = (q_coords_broadcast - k_coords_broadcast)
        + (k_size as f32 - 1.) * (q_size as f32 / k_size as f32).max(1.0);

    // Manual gathering since advanced indexing is not available
    let relative_coords_data = relative_coords.to_data();
    let rel_pos_resized_data = rel_pos_resized.to_data();
    let embedding_dim = rel_pos_resized.dims()[1];

    let mut result = Vec::with_capacity(q_size * k_size * embedding_dim);
    // Iterate in the correct order for shape [q_size, k_size, embedding_dim]
    // The flattened index for [i, j, k] is: i * (k_size * embedding_dim) + j * embedding_dim + k
    for i in 0..q_size {
        for j in 0..k_size {
            // Access element [i, j] from the 2D tensor
            let coord_idx = i * k_size + j;
            let idx = relative_coords_data.as_slice::<B::FloatElem>().unwrap()[coord_idx]
                .elem::<f32>()
                .round() as usize;
            let idx = idx.min(max_rel_dist - 1);

            // Gather all embedding dimensions for this [i, j] position
            for k in 0..embedding_dim {
                let val = rel_pos_resized_data.as_slice::<B::FloatElem>().unwrap()
                    [idx * embedding_dim + k];
                result.push(val);
            }
        }
    }

    Tensor::from_data(
        TensorData::new(result, [q_size, k_size, embedding_dim]),
        &device,
    )
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
        python.almost_equal(output, 0.1);
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
        python.almost_equal(output, 1.);
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
        python.almost_equal(output, 5.);
    }
}
