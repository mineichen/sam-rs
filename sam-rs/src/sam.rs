use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::{Bool, Float, Int};

use crate::burn_helpers::{TensorHelpers, ToFloat};
use crate::{
    modeling::{
        image_encoder::ImageEncoderViT, mask_decoder::MaskDecoder, prompt_encoder::PromptEncoder,
    },
    sam_predictor::{ImageFormat, Size},
};

#[derive(Debug, Module)]
pub struct Sam<B: Backend> {
    pub image_encoder: ImageEncoderViT<B>,
    pub prompt_encoder: PromptEncoder<B>,
    pub mask_decoder: MaskDecoder<B>,
    #[module(ignore)]
    pub pixel_mean: [f32; 3],
    #[module(ignore)]
    pub pixel_std: [f32; 3],
    #[module(ignore)]
    pub mask_threshold: f32,
    #[module(ignore)]
    pub image_format: ImageFormat,
}
#[derive(Debug)]
pub struct Input<B: Backend> {
    pub image: Tensor<B, 3, Int>,
    pub original_size: Size,
    pub boxes: Option<Tensor<B, 2>>,
    pub points: Option<(Tensor<B, 3>, Tensor<B, 2>)>,
    pub mask_inputs: Option<Tensor<B, 4>>,
}
pub struct Output<B: Backend> {
    pub masks: Tensor<B, 4, Bool>,
    pub mask_values: Tensor<B, 4, Float>,
    pub iou_predictions: Tensor<B, 2, Float>,
    pub low_res_logits: Option<Tensor<B, 4, Float>>,
    pub input_images: Tensor<B, 4, Float>,
    pub image_embeddings: Tensor<B, 4, Float>,
    pub curr_embedding: Tensor<B, 4, Float>,
}
impl<B: Backend> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    /// # SAM predicts object masks from an image and input prompts.
    ///
    /// Arguments:
    ///   - image_encoder (ImageEncoderViT): The backbone used to encode the
    ///     image into image embeddings that allow for efficient mask prediction.
    ///   - prompt_encoder (PromptEncoder): Encodes various types of input prompts.
    ///   - mask_decoder (MaskDecoder): Predicts masks from the image embeddings
    ///     and encoded prompts.
    ///   - pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
    ///   - pixel_std (list(float)): Std values for normalizing pixels in the input image.
    pub fn new(
        image_encoder: ImageEncoderViT<B>,
        prompt_encoder: PromptEncoder<B>,
        mask_decoder: MaskDecoder<B>,
        pixel_mean: Option<[f32; 3]>,
        pixel_std: Option<[f32; 3]>,
    ) -> Self {
        let pixel_mean = pixel_mean.unwrap_or([123.675, 116.28, 103.53]);
        let pixel_std = pixel_std.unwrap_or([58.395, 57.12, 57.375]);

        Self {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            pixel_mean,
            pixel_std,
            mask_threshold: 0.0,
            image_format: ImageFormat::RGB,
        }
    }
    fn pixel_mean(&self, device: &B::Device) -> Tensor<B, 3> {
        Tensor::of_slice(self.pixel_mean.to_vec(), [self.pixel_mean.len()], device).reshape_max([
            usize::MAX,
            1,
            1,
        ])
    }
    fn pixel_std(&self, device: &B::Device) -> Tensor<B, 3> {
        Tensor::of_slice(self.pixel_std.to_vec(), [self.pixel_std.len()], device).reshape_max([
            usize::MAX,
            1,
            1,
        ])
    }

    /// Predicts masks end-to-end from provided images and prompts.
    /// If prompts are not known in advance, using SamPredictor is
    /// recommended over calling the model directly.
    ///
    /// Arguments:
    ///   - batched_input (list(dict)): A list over input images, each a
    ///     dictionary with the following keys. A prompt key can be
    ///     excluded if it is not present.
    ///       - `image`: The image as a torch tensor in 3xHxW format,
    ///         already transformed for input to the model.
    ///       - `original_size`: (tuple(int, int)) The original size of
    ///         the image before transformation, as (H, W).
    ///       - `point_coords`: (torch.Tensor) Batched point prompts for
    ///         this image, with shape BxNx2. Already transformed to the
    ///         input frame of the model.
    ///       - `point_labels`: (torch.Tensor) Batched labels for point prompts,
    ///         with shape BxN.
    ///       - `boxes`: (torch.Tensor) Batched box inputs, with shape Bx4.
    ///         Already transformed to the input frame of the model.
    ///       - `mask_inputs`: (torch.Tensor) Batched mask inputs to the model,
    ///         in the form Bx1xHxW.
    ///   - multimask_output (bool): Whether the model should predict multiple
    ///     disambiguating masks, or return a single mask.
    ///
    /// Returns:
    ///   (list(dict)): A list over input images, where each element is
    ///     as dictionary with the following keys.
    ///       - `masks`: (torch.Tensor) Batched binary mask predictions,
    ///         with shape BxCxHxW, where B is the number of input prompts,
    ///         C is determined by multimask_output, and (H, W) is the
    ///         original size of the image.
    ///       - `iou_predictions`: (torch.Tensor) The model's predictions
    ///         of mask quality, in shape BxC.
    ///       - `low_res_logits`: (torch.Tensor) Low resolution logits with
    ///         shape BxCxHxW, where H=W=256. Can be passed as mask input
    ///         to subsequent iterations of prediction.
    pub fn forward(
        &mut self,
        batched_input: Vec<Input<B>>,
        multimask_output: bool,
        device: &B::Device,
    ) -> Vec<Output<B>> {
        let processed_images = batched_input
            .iter()
            .map(|x| self.preprocess(x.image.clone()))
            .collect::<Vec<_>>();
        let input_images = Tensor::stack(processed_images.clone(), 0);
        let image_embeddings = self.image_encoder.forward(input_images.clone());
        // TODO: Implement proper tensor unbinding when available in Burn 0.18.0
        // For now, create a single-element vector as a workaround
        let image_embeddings_vec: Vec<Tensor<B, 4>> = vec![image_embeddings.clone()]; // Simplified workaround

        assert_eq!(image_embeddings_vec.len(), batched_input.len());
        let mut outputs: Vec<Output<B>> = vec![];
        for (image_record, curr_embedding) in batched_input.iter().zip(image_embeddings_vec) {
            let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                image_record.points.clone(),
                image_record.boxes.clone(),
                image_record.mask_inputs.clone(),
                device,
            );
            let image_pe = self.prompt_encoder.get_dense_pe();
            let (low_res_masks, iou_predictions) = self.mask_decoder.forward(
                curr_embedding.clone().unsqueeze(),
                image_pe.clone(),
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
            );
            let size = image_record.image.dims();
            let mask_values = self.postprocess_masks(
                low_res_masks.clone(),
                Size(size[size.len() - 2], size[size.len() - 1]),
                image_record.original_size,
            );
            let masks = mask_values.clone().greater_elem(self.mask_threshold);
            outputs.push(Output {
                masks,
                mask_values,
                input_images: input_images.clone(),
                image_embeddings: image_embeddings.clone(),
                curr_embedding: curr_embedding.clone(),
                iou_predictions,
                low_res_logits: Some(low_res_masks),
            })
        }
        outputs
    }

    /// Remove padding and upscale masks to the original image size.
    /// Arguments:
    ///   masks (torch.Tensor): Batched masks from the mask_decoder,
    ///     in BxCxHxW format.
    ///   input_size (tuple(int, int)): The size of the image input to the
    ///     model, in (H, W) format. Used to remove padding.
    ///   original_size (tuple(int, int)): The original size of the image
    ///     before resizing for input to the model, in (H, W) format.
    /// Returns:
    ///   (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
    ///     is given by original_size.
    pub fn postprocess_masks(
        &self,
        masks: Tensor<B, 4, Float>,
        input: Size,
        original: Size,
    ) -> Tensor<B, 4, Float> {
        let output_size = self.image_encoder.img_size;
        // Upsample masks to output_size (e.g., 1024x1024)
        let masks = self.bilinear_upsample(masks, output_size, output_size);
        // Remove padding
        let masks: Tensor<B, 4> = masks.narrow(2, 0, input.0);
        let masks = masks.narrow(3, 0, input.1);
        // Upsample to original size
        let masks = self.bilinear_upsample(masks, original.0, original.1);
        masks
    }

    fn bilinear_upsample(
        &self,
        image: Tensor<B, 4>,
        target_h: usize,
        target_w: usize,
    ) -> Tensor<B, 4> {
        use burn::tensor::{ElementConversion, TensorData};

        let shape = image.dims();
        let (batch, channels, src_h, src_w) = (shape[0], shape[1], shape[2], shape[3]);

        let device = image.device();
        let image_data = image.to_data();
        let image_slice = image_data.as_slice::<B::FloatElem>().unwrap();

        let mut result = Vec::with_capacity(batch * channels * target_h * target_w);

        // F.interpolate with mode='bilinear', align_corners=False
        for b in 0..batch {
            for c in 0..channels {
                for i in 0..target_h {
                    for j in 0..target_w {
                        let src_y =
                            ((i as f32 + 0.5) * src_h as f32 / target_h as f32 - 0.5).max(0.0);
                        let src_x =
                            ((j as f32 + 0.5) * src_w as f32 / target_w as f32 - 0.5).max(0.0);

                        let y0 = src_y.floor() as usize;
                        let y1 = (y0 + 1).min(src_h - 1);
                        let x0 = src_x.floor() as usize;
                        let x1 = (x0 + 1).min(src_w - 1);

                        let wy = src_y - y0 as f32;
                        let wx = src_x - x0 as f32;

                        let v00 = image_slice
                            [b * channels * src_h * src_w + c * src_h * src_w + y0 * src_w + x0]
                            .elem::<f32>();
                        let v01 = image_slice
                            [b * channels * src_h * src_w + c * src_h * src_w + y0 * src_w + x1]
                            .elem::<f32>();
                        let v10 = image_slice
                            [b * channels * src_h * src_w + c * src_h * src_w + y1 * src_w + x0]
                            .elem::<f32>();
                        let v11 = image_slice
                            [b * channels * src_h * src_w + c * src_h * src_w + y1 * src_w + x1]
                            .elem::<f32>();

                        let interp = v00 * (1.0 - wx) * (1.0 - wy)
                            + v01 * wx * (1.0 - wy)
                            + v10 * (1.0 - wx) * wy
                            + v11 * wx * wy;

                        result.push(B::FloatElem::from_elem(interp));
                    }
                }
            }
        }

        Tensor::from_data(
            TensorData::new(result, [batch, channels, target_h, target_w]),
            &device,
        )
    }

    /// Normalize pixel values and pad to a square input.
    pub fn preprocess<const D: usize>(&self, x: Tensor<B, D, Int>) -> Tensor<B, D, Float> {
        let device = x.device();

        #[cfg(test)]
        {
            println!("Preprocess input shape: {:?}", x.shape());
            let pm = self.pixel_mean(&device);
            let ps = self.pixel_std(&device);
            println!(
                "Pixel mean shape: {:?}, values: {:?}",
                pm.shape(),
                self.pixel_mean
            );
            println!(
                "Pixel std shape: {:?}, values: {:?}",
                ps.shape(),
                self.pixel_std
            );
            println!(
                "Pixel mean unsqueezed shape: {:?}",
                pm.unsqueeze::<D>().shape()
            );
        }

        let x: Tensor<B, D, Float> = (x.to_float() - self.pixel_mean(&device).unsqueeze())
            / self.pixel_std(&device).unsqueeze();
        let size = x.dims();
        let (h, w) = (size[D - 2], size[D - 1]);

        let padh = self.image_encoder.img_size - h;
        let padw = self.image_encoder.img_size - w;
        // In Burn 0.18.0, pad takes (left, right, top, bottom) for 2D padding
        let x = x.pad((0, padw, 0, padh), 0.);
        x
    }
}

#[cfg(test)]
mod test {
    use pyo3::types::{PyAnyMethods, PyDictMethods, PyListMethods};
    use pyo3::{
        types::{PyDict, PyList},
        Python,
    };

    use crate::{
        python::python_data::{random_python_tensor, random_python_tensor_int, PythonData},
        tests::helpers::{get_python_test_sam, get_test_sam},
    };

    use super::Input;

    #[test]
    fn test_sam_forward_boxes() {
        let original_size = (100, 200);
        Python::attach(|py| {
            let sam = get_python_test_sam(&py)?;
            let map = crate::python::recorder::get_python_map(sam.clone())?;

            let image = random_python_tensor_int(py, [3, 8, 8])?;
            let boxes = random_python_tensor(py, [4, 4])?;
            let image_data: PythonData<3> = image.clone().try_into()?;
            let boxes_data: PythonData<2> = boxes.clone().try_into()?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("image", &image)?;
            kwargs.set_item("boxes", &boxes)?;
            kwargs.set_item("original_size", original_size)?;

            let output = sam
                .call1(([kwargs], false))?
                .cast::<PyList>()?
                .get_item(0)?;
            let mask_values: PythonData<4> = output.get_item("mask_values")?.try_into()?;
            let iou_predictions: PythonData<2> = output.get_item("iou_predictions")?.try_into()?;
            let low_res_logits: PythonData<4> = output.get_item("low_res_logits")?.try_into()?;

            let device = Default::default();
            let mut rust_sam = crate::python::recorder::load_sam(get_test_sam(&device), map);
            let input = Input {
                image: image_data.into(),
                boxes: Some(boxes_data.into()),
                original_size: original_size.into(),
                mask_inputs: None,
                points: None,
            };
            let rust_output = rust_sam.forward(vec![input], false, &device);
            let rust_output = rust_output.get(0).unwrap();

            mask_values.almost_equal(rust_output.mask_values.clone(), 2e-3);
            iou_predictions.almost_equal(rust_output.iou_predictions.clone(), None);
            low_res_logits.almost_equal(rust_output.low_res_logits.clone().unwrap(), 3e-3);

            Ok::<_, pyo3::PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn test_sam_forward_points() {
        let original_size = (100, 200);
        Python::attach(|py| {
            let sam = get_python_test_sam(&py)?;
            let map = crate::python::recorder::get_python_map(sam.clone())?;

            let image = random_python_tensor_int(py, [3, 8, 8])?;
            let points = random_python_tensor(py, [4, 2, 2])?;
            let labels = random_python_tensor(py, [4, 2])?;
            let image_data: PythonData<3> = image.clone().try_into()?;
            let points_data: PythonData<3> = points.clone().try_into()?;
            let labels_data: PythonData<2> = labels.clone().try_into()?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("image", image)?;
            kwargs.set_item("point_coords", points)?;
            kwargs.set_item("point_labels", labels)?;
            kwargs.set_item("original_size", original_size)?;

            let output = sam
                .call1(([kwargs], false))?
                .cast::<PyList>()?
                .get_item(0)?;
            let mask_values: PythonData<4> = output.get_item("mask_values")?.try_into()?;
            let iou_predictions: PythonData<2> = output.get_item("iou_predictions")?.try_into()?;
            let low_res_logits: PythonData<4> = output.get_item("low_res_logits")?.try_into()?;

            let device = Default::default();
            let mut rust_sam = crate::python::recorder::load_sam(get_test_sam(&device), map);
            let input = Input {
                image: image_data.into(),
                boxes: None,
                original_size: original_size.into(),
                mask_inputs: None,
                points: Some((points_data.into(), labels_data.into())),
            };
            let rust_output = rust_sam.forward(vec![input], false, &device);
            let rust_output = rust_output.get(0).unwrap();

            mask_values.almost_equal(rust_output.mask_values.clone(), None);
            iou_predictions.almost_equal(rust_output.iou_predictions.clone(), None);
            low_res_logits.almost_equal(rust_output.low_res_logits.clone().unwrap(), None);

            Ok::<_, pyo3::PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn test_sam_postprocess_masks() {
        let input_size = (684, 1024);
        let original = (534, 800);
        Python::attach(|py| {
            let sam = get_python_test_sam(&py)?;
            let masks = random_python_tensor(py, [4, 1, 256, 256])?;
            let masks_data: PythonData<4> = masks.clone().try_into()?;
            let output = sam.call_method1("postprocess_masks", (masks, input_size, original))?;
            let output_data: PythonData<4> = output.try_into()?;

            let device = Default::default();
            let rust_sam = get_test_sam(&device);
            let rust_output =
                rust_sam.postprocess_masks(masks_data.into(), input_size.into(), original.into());

            output_data.almost_equal(rust_output, None);
            Ok::<_, pyo3::PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn test_sam_preprocess() {
        Python::attach(|py| {
            let sam = get_python_test_sam(&py)?;
            let input = random_python_tensor_int(py, [3, 171, 128])?;
            let input_data: PythonData<3> = input.clone().try_into()?;
            let output = sam.call_method1("preprocess", (input,))?;
            let output_data: PythonData<3> = output.try_into()?;

            let device = Default::default();
            let rust_sam = get_test_sam(&device);
            let rust_output = rust_sam.preprocess(input_data.into());

            output_data.almost_equal(rust_output, None);
            Ok::<_, pyo3::PyErr>(())
        })
        .unwrap();
    }
}
