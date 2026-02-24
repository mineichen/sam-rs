use std::f32::consts::PI;

use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

use crate::sam_predictor::Size;

/// Positional encoding using random spatial frequencies.
#[derive(Debug, Module)]
pub struct PositionEmbeddingRandom<B: Backend> {
    // Store the positional encoding matrix as a learned parameter (loaded from checkpoint)
    pub positional_encoding_gaussian_matrix: Param<Tensor<B, 2>>,
}

impl<B: Backend> PositionEmbeddingRandom<B> {
    pub fn new(num_pos_feats: Option<usize>, scale: Option<f32>, device: &B::Device) -> Self {
        let num_pos_feats = num_pos_feats.unwrap_or(64);
        let mut scale = scale.unwrap_or(1.0);

        if scale <= 0.0 {
            scale = 1.0;
        }

        // Generate the positional encoding matrix once during initialization
        // Python uses torch.ones scaled, not random (despite the class name)
        let positional_encoding_gaussian_matrix =
            Param::from_tensor(Tensor::ones([2, num_pos_feats], device).mul_scalar(scale));

        Self {
            positional_encoding_gaussian_matrix,
        }
    }
    ///Positionally encode points that are normalized to [0,1].
    fn _pe_encoding(&self, coords: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut coords = coords.mul_scalar(2.0) - 1.0;
        coords = coords.matmul(
            self.positional_encoding_gaussian_matrix
                .val()
                .clone()
                .unsqueeze(),
        );
        coords = coords.mul_scalar(2.0 * PI);
        Tensor::cat(vec![coords.clone().sin(), coords.cos()], 2)
    }

    /// Generate positional encoding for a grid of the specified size.
    pub fn forward(&self, size: Size) -> Tensor<B, 3> {
        let Size(h, w) = size;
        let device = Default::default();

        // y_embed: values increase along height dimension [h, w]
        // y_embed[i, j] = (i + 0.5) / h
        let y_embed: Tensor<B, 2> = Tensor::arange(0..h as i64, &device)
            .float()
            .add_scalar(0.5)
            .reshape([h, 1])
            .repeat_dim(1, w)
            .div_scalar(h as f32);

        // x_embed: values increase along width dimension [h, w]
        // x_embed[i, j] = (j + 0.5) / w
        let x_embed: Tensor<B, 2> = Tensor::arange(0..w as i64, &device)
            .float()
            .add_scalar(0.5)
            .reshape([1, w])
            .repeat_dim(0, h)
            .div_scalar(w as f32);

        let pe: Tensor<B, 3> = self._pe_encoding(Tensor::stack(vec![x_embed, y_embed], 2));
        pe.permute([2, 0, 1])
    }

    /// Positionally encode points that are not normalized to [0,1].
    pub fn forward_with_coords(&self, coords: Tensor<B, 3>, image_size: Size) -> Tensor<B, 3> {
        // Normalize coordinates to [0, 1]
        // coords[..., 0] = coords[..., 0] / image_size.1
        // coords[..., 1] = coords[..., 1] / image_size.0

        let coords_x = coords
            .clone()
            .narrow(2, 0, 1)
            .div_scalar(image_size.1 as f32);
        let coords_y = coords
            .clone()
            .narrow(2, 1, 1)
            .div_scalar(image_size.0 as f32);

        let normalized_coords = Tensor::cat(vec![coords_x, coords_y], 2);
        self._pe_encoding(normalized_coords)
    }
}

#[cfg(test)]
mod test {
    use burn::tensor::backend::Backend;
    use pyo3::types::PyAnyMethods;
    use pyo3::{PyResult, Python};

    use crate::{
        python::python_data::{random_python_tensor, PythonData},
        sam_predictor::Size,
        tests::helpers::TestBackend,
    };

    #[test]
    fn test_position_embedding_pe_encoding() {
        fn python() -> PyResult<(PythonData<3>, PythonData<3>)> {
            Python::attach(|py| {
                let module = py
                    .import("segment_anything.modeling.prompt_encoder")?
                    .getattr("PositionEmbeddingRandom")?;
                let module = module.call1((128,))?;

                let input = random_python_tensor(py, [64, 69, 2])?;
                let output = module.call_method1("_pe_encoding", (input.clone(),))?;
                Ok((input.try_into()?, output.try_into()?))
            })
        }
        let (input, python) = python().unwrap();
        let device: <TestBackend as Backend>::Device = Default::default();
        let pos_embedding: super::PositionEmbeddingRandom<TestBackend> =
            super::PositionEmbeddingRandom::new(Some(128), None, &device);

        let output = pos_embedding._pe_encoding(input.into());
        python.almost_equal(output, None);
    }

    #[test]
    fn test_position_embedding_forward() {
        fn python() -> PyResult<PythonData<3>> {
            Python::attach(|py| {
                let module = py
                    .import("segment_anything.modeling.prompt_encoder")?
                    .getattr("PositionEmbeddingRandom")?;
                let module = module.call1((128,))?;

                let output = module.call_method1("forward", ((64, 64),))?;
                Ok(output.try_into()?)
            })
        }
        let python = python().unwrap();
        let device: <TestBackend as Backend>::Device = Default::default();
        let pos_embedding: super::PositionEmbeddingRandom<TestBackend> =
            super::PositionEmbeddingRandom::new(Some(128), None, &device);

        let output = pos_embedding.forward(Size(64, 64));
        python.almost_equal(output, None);
    }

    #[test]
    fn test_position_embedding_with_coords() {
        fn python() -> PyResult<(PythonData<3>, PythonData<3>)> {
            Python::attach(|py| {
                let module = py
                    .import("segment_anything.modeling.prompt_encoder")?
                    .getattr("PositionEmbeddingRandom")?;
                let module = module.call1((128,))?;
                let input = random_python_tensor(py, [64, 2, 2])?;
                let output = module
                    .getattr("forward_with_coords")?
                    .call1((input.clone(), (1024, 1024)))?;
                Ok((input.try_into()?, output.try_into()?))
            })
        }
        let (input, python) = python().unwrap();
        let device: <TestBackend as Backend>::Device = Default::default();
        let pos_embedding: super::PositionEmbeddingRandom<TestBackend> =
            super::PositionEmbeddingRandom::new(Some(128), None, &device);
        let output = pos_embedding.forward_with_coords(input.into(), Size(1024, 1024));
        python.almost_equal(output, None);
    }
}

// Note: In Burn 0.18.0, the Module trait has changed
// The forward method is not part of the Module trait anymore
// This struct doesn't need to implement Module as it's not a neural network layer
