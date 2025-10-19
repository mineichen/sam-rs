mod positional_embedding;

use self::positional_embedding::PositionEmbeddingRandom;
use super::common::{activation::Activation, layer_norm_2d::LayerNorm2d};
use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};
use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Embedding, EmbeddingConfig,
    },
    tensor::{backend::Backend, Bool, Tensor},
};

#[derive(Debug, Module)]
pub struct PromptEncoder<B: Backend> {
    pub embed_dim: usize,
    pub input_image_size: Size,
    pub image_embedding_size: Size,
    pub pe_layer: PositionEmbeddingRandom<B>,
    pub point_embeddings: Vec<Embedding<B>>,
    pub no_mask_embed: Embedding<B>,
    pub not_a_point_embed: Embedding<B>,
    pub mask_downscaling0: Conv2d<B>,
    pub mask_downscaling1: LayerNorm2d<B>,
    pub mask_downscaling2: Activation,
    pub mask_downscaling3: Conv2d<B>,
    pub mask_downscaling4: LayerNorm2d<B>,
    pub mask_downscaling5: Activation,
    pub mask_downscaling6: Conv2d<B>,
}

impl<B: Backend> PromptEncoder<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    // Encodes prompts for input to SAM's mask decoder.
    // Arguments:
    //   embed_dim (int): The prompts' embedding dimension
    //   image_embedding_size (tuple(int, int)): The spatial size of the
    //     image embedding, as (H, W).
    //   input_image_size (int): The padded size of the image as input
    //     to the image encoder, as (H, W).
    //   mask_in_chans (int): The number of hidden channels used for
    //     encoding input masks.
    //   activation (nn.Module): The activation to use when encoding
    //     input masks.
    pub fn new(
        embed_dim: usize,
        image_embedding_size: Size,
        input_image_size: Size,
        mask_in_chans: usize,
        activation: Option<Activation>,
        device: &B::Device,
    ) -> Self {
        let activation = activation.unwrap_or(Activation::GELU);

        let pe_layer = PositionEmbeddingRandom::new(Some(embed_dim / 2), None, device);
        let num_point_embeddings: usize = 4; // pos/neg point + 2 box corners
        let mut point_embeddings = vec![];
        for _ in 0..num_point_embeddings {
            point_embeddings.push(EmbeddingConfig::new(1, embed_dim).init(device));
        }
        let not_a_point_embed = EmbeddingConfig::new(1, embed_dim).init(device);

        let mask_downscaling0 = Conv2dConfig::new([1, mask_in_chans / 4], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let mask_downscaling1 = LayerNorm2d::new(mask_in_chans / 4, None, device);
        let mask_downscaling2 = activation;
        let mask_downscaling3 = Conv2dConfig::new([mask_in_chans / 4, mask_in_chans], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let mask_downscaling4 = LayerNorm2d::new(mask_in_chans, None, device);
        let mask_downscaling5 = activation;
        let mask_downscaling6 = Conv2dConfig::new([mask_in_chans, embed_dim], [1, 1]).init(device);

        let no_mask_embed = EmbeddingConfig::new(1, embed_dim).init(device);
        Self {
            embed_dim,
            input_image_size,
            image_embedding_size,
            pe_layer,
            point_embeddings,
            no_mask_embed,
            not_a_point_embed,
            mask_downscaling0,
            mask_downscaling1,
            mask_downscaling2,
            mask_downscaling3,
            mask_downscaling4,
            mask_downscaling5,
            mask_downscaling6,
        }
    }

    // Returns the positional encoding used to encode point prompts,
    // applied to a dense set of points the shape of the image encoding.

    // Returns:
    //   torch.Tensor: Positional encoding with shape
    //     1x(embed_dim)x(embedding_h)x(embedding_w)
    pub fn get_dense_pe(&self) -> Tensor<B, 4> {
        return self.pe_layer.forward(self.image_embedding_size).unsqueeze();
    }

    /// Embeds point prompts.
    fn _embed_points(&self, points: Tensor<B, 3>, labels: Tensor<B, 2>, pad: bool) -> Tensor<B, 3> {
        let mut points = points + 0.5; // Shift to center of pixel
        let mut labels = labels;
        let device = points.device();
        if pad {
            let padding_point = Tensor::zeros([points.dims()[0], 1, 2], &device);
            let padding_label = -Tensor::ones([labels.dims()[0], 1], &device);
            points = Tensor::cat(vec![points, padding_point], 1);
            labels = Tensor::cat(vec![labels, padding_label], 1);
        }
        let mut point_embedding = self
            .pe_layer
            .forward_with_coords(points, self.input_image_size);

        let mask_minus_one: Tensor<B, 3, Bool> = labels.clone().equal_elem(-1.).unsqueeze_end();
        let mask_zero: Tensor<B, 3, Bool> = labels.clone().equal_elem(0.).unsqueeze_end();
        let mask_one: Tensor<B, 3, Bool> = labels.clone().equal_elem(1.).unsqueeze_end();

        // In Burn 0.18.0, implement conditional selection manually
        // Original logic:
        // 1. Zero out point_embedding where label == -1
        // 2. Add not_a_point_embed where label == -1
        // 3. Add point_embeddings[0] where label == 0
        // 4. Add point_embeddings[1] where label == 1

        // Convert boolean masks to float for multiplication
        let mask_minus_one_float = mask_minus_one.clone().float();
        let mask_zero_float = mask_zero.clone().float();
        let mask_one_float = mask_one.clone().float();

        // Step 1: Zero out where label == -1
        // point_embedding * (1 - mask_minus_one) keeps values where mask is false, zeros where true
        point_embedding = point_embedding
            * (Tensor::ones_like(&mask_minus_one_float) - mask_minus_one_float.clone());

        // Step 2: Add not_a_point_embed where label == -1
        let not_a_point_embed_weight = self
            .not_a_point_embed
            .clone()
            .into_record()
            .weight
            .val()
            .unsqueeze();
        point_embedding =
            point_embedding.clone() + (not_a_point_embed_weight * mask_minus_one_float);

        // Step 3: Add point_embeddings[0] where label == 0
        let point_embed_0_weight = self.point_embeddings[0]
            .clone()
            .into_record()
            .weight
            .val()
            .unsqueeze();
        point_embedding = point_embedding.clone() + (point_embed_0_weight * mask_zero_float);

        // Step 4: Add point_embeddings[1] where label == 1
        let point_embed_1_weight = self.point_embeddings[1]
            .clone()
            .into_record()
            .weight
            .val()
            .unsqueeze();
        point_embedding = point_embedding + (point_embed_1_weight * mask_one_float);

        point_embedding
    }

    ///Embeds box prompts.
    fn _embed_boxes(&self, boxes: Tensor<B, 2>) -> Tensor<B, 3> {
        let boxes = boxes + 0.5; // Shift to center of pixel
        let coords = boxes.reshape_max([usize::MAX, 2, 2]);
        let mut corner_embedding = self
            .pe_layer
            .forward_with_coords(coords, self.input_image_size);

        let corner_embedding_0 = corner_embedding.clone().narrow(1, 0, 1);
        let corner_embedding_1 = corner_embedding.narrow(1, 1, 1);

        let updated_corner_embedding_0 = corner_embedding_0
            + self.point_embeddings[2]
                .clone()
                .into_record()
                .weight
                .val()
                .unsqueeze();
        let updated_corner_embedding_1 = corner_embedding_1
            + self.point_embeddings[3]
                .clone()
                .into_record()
                .weight
                .val()
                .unsqueeze();

        corner_embedding = Tensor::cat(
            vec![updated_corner_embedding_0, updated_corner_embedding_1],
            1,
        );

        corner_embedding
    }

    ///Embeds mask inputs.
    fn _embed_masks(&self, masks: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut masks = masks;
        masks = self.mask_downscaling0.forward(masks);
        masks = self.mask_downscaling1.forward(masks);
        masks = self.mask_downscaling2.forward(masks);
        masks = self.mask_downscaling3.forward(masks);
        masks = self.mask_downscaling4.forward(masks);
        masks = self.mask_downscaling5.forward(masks);
        masks = self.mask_downscaling6.forward(masks);
        masks
    }

    /// Gets the batch size of the output given the batch size of the input prompts.
    fn _get_batch_size(
        &self,
        points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
        boxes: Option<Tensor<B, 2>>,
        masks: Option<Tensor<B, 4>>,
    ) -> usize {
        if let Some((point, _)) = points {
            return point.dims()[0];
        } else if let Some(boxes) = boxes {
            return boxes.dims()[0];
        } else if let Some(masks) = masks {
            return masks.dims()[0];
        } else {
            return 1;
        }
    }

    // Embeds different types of prompts, returning both sparse and dense
    // embeddings.
    // Arguments:
    //   points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
    //     and labels to embed.
    //   boxes (torch.Tensor or none): boxes to embed
    //   masks (torch.Tensor or none): masks to embed
    // Returns:
    //   torch.Tensor: sparse embeddings for the points and boxes, with shape
    //     BxNx(embed_dim), where N is determined by the number of input points
    //     and boxes.
    //   torch.Tensor: dense embeddings for the masks, in the shape
    //     Bx(embed_dim)x(embed_H)x(embed_W)
    pub fn forward(
        &self,
        points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
        boxes: Option<Tensor<B, 2>>,
        masks: Option<Tensor<B, 4>>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let bs = self._get_batch_size(points.clone(), boxes.clone(), masks.clone());
        let mut sparse_embeddings = Tensor::empty([bs, 0, self.embed_dim], device);
        if let Some((coords, labels)) = points {
            let point_embeddings = self._embed_points(coords, labels, boxes.is_none());
            sparse_embeddings = Tensor::cat(vec![sparse_embeddings, point_embeddings], 1);
        }
        if let Some(boxes) = boxes {
            let box_embeddings = self._embed_boxes(boxes);
            sparse_embeddings = Tensor::cat(vec![sparse_embeddings, box_embeddings], 1);
        }
        let dense_embeddings = match masks {
            Some(masks) => self._embed_masks(masks),
            None => {
                // Get the no_mask_embed weight and reshape to [1, embed_dim, 1, 1]
                let no_mask = self
                    .no_mask_embed
                    .clone()
                    .into_record()
                    .weight
                    .val()
                    .reshape_max([1, usize::MAX, 1, 1]);

                // Expand to [bs, embed_dim, image_embedding_size.0, image_embedding_size.1]
                // In Burn 0.18.0, use repeat_dim instead of expand
                let no_mask = no_mask
                    .repeat_dim(0, bs) // Repeat batch dimension
                    .repeat_dim(2, self.image_embedding_size.0) // Repeat height dimension
                    .repeat_dim(3, self.image_embedding_size.1); // Repeat width dimension
                no_mask
            }
        };
        (sparse_embeddings, dense_embeddings)
    }
}

#[cfg(test)]
mod test {

    use burn::prelude::Backend;
    use pyo3::types::PyAnyMethods;
    use pyo3::{types::PyTuple, PyAny, PyResult, Python};

    use crate::{
        modeling::common::activation::Activation,
        python::{
            module_to_file::module_to_file,
            python_data::{random_python_tensor, PythonData},
        },
        sam_predictor::Size,
        tests::helpers::{load_module, TestBackend},
    };

    use super::PromptEncoder;
    const MASK_IN_CHANS: usize = 8;
    const EMBED_DIM: usize = 128;

    fn _init(device: &<TestBackend as Backend>::Device) -> PromptEncoder<TestBackend> {
        let prompt_encoder = PromptEncoder::new(
            EMBED_DIM,
            Size(32, 32),
            Size(512, 512),
            MASK_IN_CHANS,
            Some(Activation::GELU),
            device,
        );
        prompt_encoder
    }
    fn get_python_module<'a>(py: &'a Python, file: &str) -> PyResult<pyo3::Bound<'a, PyAny>> {
        let gelu = py.import("torch.nn")?.getattr("GELU")?;
        let module = py
            .import("segment_anything.modeling.prompt_encoder")?
            .getattr("PromptEncoder")?;
        let module = module.call1((128, (32, 32), (512, 512), 8, gelu))?;
        module_to_file(file, *py, &module)?;
        Ok(module)
    }
    fn python_embed_points(
        file: &str,
        with_pad: bool,
    ) -> PyResult<(PythonData<3>, PythonData<2>, PythonData<3>)> {
        Python::attach(|py| {
            let module = get_python_module(&py, file)?;

            let points = random_python_tensor(py, [32, 1, 2])?;
            let labels = random_python_tensor(py, [32, 1])?;
            let output = module
                .getattr("_embed_points")?
                .call1((&points, &labels, with_pad))?;
            Ok((points.try_into()?, labels.try_into()?, output.try_into()?))
        })
    }
    #[test]
    fn test_prompt_encoder_embed_points_pad() {
        const FILE: &str = "prompt_encoder_embed_points_pad";
        let (points, labels, python) = python_embed_points(FILE, true).unwrap();
        let device = Default::default();
        let mut prompt_encoder = _init(&device);
        prompt_encoder = load_module(FILE, prompt_encoder);

        let output = prompt_encoder._embed_points(points.into(), labels.into(), true);
        python.almost_equal(output, None);
    }
    #[test]
    fn test_prompt_encoder_embed_points_no_pad() {
        const FILE: &str = "prompt_encoder_embed_points_no_pad";
        let (points, labels, python) = python_embed_points(FILE, false).unwrap();
        let device = Default::default();
        let mut prompt_encoder = _init(&device);
        prompt_encoder = load_module(FILE, prompt_encoder);

        let output = prompt_encoder._embed_points(points.into(), labels.into(), false);
        python.almost_equal(output, None);
    }

    #[test]
    fn test_prompt_encoder_embed_boxes() {
        const FILE: &str = "prompt_encoder_embed_boxes";
        fn python() -> PyResult<(PythonData<2>, PythonData<3>)> {
            Python::attach(|py| {
                let module = get_python_module(&py, FILE)?;

                let boxes = random_python_tensor(py, [32, 4])?;
                let output = module.call_method1("_embed_boxes", (boxes.clone(),))?;
                Ok((boxes.try_into()?, output.try_into()?))
            })
        }
        let (boxes, python) = python().unwrap();
        let device = Default::default();
        let mut prompt_encoder = _init(&device);
        prompt_encoder = load_module(FILE, prompt_encoder);

        let output = prompt_encoder._embed_boxes(boxes.into());
        python.almost_equal(output, None);
    }

    #[test]
    fn test_prompt_encoder_embed_masks() {
        const FILE: &str = "prompt_encoder_embed_masks";
        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::attach(|py| {
                let module = get_python_module(&py, FILE)?;

                let masks = random_python_tensor(py, [8, 1, 4, 4])?;
                let output = module.call_method1("_embed_masks", (masks.clone(),))?;
                Ok((masks.try_into()?, output.try_into()?))
            })
        }
        let (masks, python) = python().unwrap();
        let device = Default::default();
        let mut prompt_encoder = _init(&device);
        prompt_encoder = load_module(FILE, prompt_encoder);

        let output = prompt_encoder._embed_masks(masks.into());
        python.almost_equal(output, None);
    }

    #[test]
    fn test_prompt_encoder_forward_points() {
        const FILE: &str = "prompt_encoder_forward_points";
        fn python() -> PyResult<(PythonData<3>, PythonData<2>, PythonData<3>, PythonData<4>)> {
            Python::attach(|py| {
                let module = get_python_module(&py, FILE)?;
                let points = random_python_tensor(py, [8, 1, 2])?;
                let labels = random_python_tensor(py, [8, 1])?;
                let output =
                    module.call_method1("forward", ((&points, &labels), py.None(), py.None()))?;
                let output = output.downcast::<PyTuple>()?;
                let sparse = output.get_item(0)?;
                let dense = output.get_item(1)?;
                Ok((
                    points.try_into()?,
                    labels.try_into()?,
                    sparse.try_into()?,
                    dense.try_into()?,
                ))
            })
        }
        let (points, labels, sparse, dense) = python().unwrap();
        let device = &Default::default();
        let mut prompt_encoder = _init(device);
        prompt_encoder = load_module(FILE, prompt_encoder);

        let (sparse2, dense2) =
            prompt_encoder.forward(Some((points.into(), labels.into())), None, None, device);
        sparse.almost_equal(sparse2, None);
        dense.almost_equal(dense2, None);
    }
    #[test]
    fn test_prompt_encoder_forward_boxes() {
        const FILE: &str = "prompt_encoder_forward_boxes";
        fn python() -> PyResult<(PythonData<2>, PythonData<3>, PythonData<4>)> {
            Python::attach(|py| {
                let module = get_python_module(&py, FILE)?;
                let boxes = random_python_tensor(py, [8, 4])?;
                let output = module
                    .getattr("forward")?
                    .call1((py.None(), &boxes, py.None()))?;
                let output = output.downcast::<PyTuple>()?;
                let sparse = output.get_item(0)?;
                let dense = output.get_item(1)?;
                Ok((boxes.try_into()?, sparse.try_into()?, dense.try_into()?))
            })
        }
        let (boxes, sparse, dense) = python().unwrap();
        let device = &Default::default();
        let mut prompt_encoder = _init(device);
        prompt_encoder = load_module(FILE, prompt_encoder);
        let (sparse2, dense2) = prompt_encoder.forward(None, Some(boxes.into()), None, device);
        sparse.almost_equal(sparse2, None);
        dense.almost_equal(dense2, None);
    }
}
