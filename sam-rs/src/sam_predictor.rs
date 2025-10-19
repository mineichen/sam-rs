use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::burn_helpers::ToFloat;
use crate::sam::Sam;
use crate::utils::transforms::ResizeLongestSide;

pub struct SamPredictor<B: Backend> {
    is_image_set: bool,
    features: Option<Tensor<B, 4>>,
    orig_h: Option<i64>,
    orig_w: Option<i64>,
    input_h: Option<i64>,
    input_w: Option<i64>,
    input_size: Option<Size>,
    original_size: Option<Size>,
    pub model: Sam<B>,
    pub transfrom: ResizeLongestSide,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Module)]
pub enum ImageFormat {
    RGB,
    BGR,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Size(pub usize, pub usize);

// Implement Module for Size so it can be used in Module-derived structs
// Size is just a configuration parameter, not a trainable module
impl<B: Backend> burn::module::Module<B> for Size {
    type Record = SizeRecord;

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // No parameters to visit
    }

    fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        SizeRecord(Some((self.0, self.1)))
    }

    fn num_params(&self) -> usize {
        0
    }
}

impl<B: burn::tensor::backend::AutodiffBackend> burn::module::AutodiffModule<B> for Size {
    type InnerModule = Size;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl burn::module::ModuleDisplay for Size {
    fn format(&self, _passed_settings: burn::module::DisplaySettings) -> String {
        format!("Size({}, {})", self.0, self.1)
    }
}

impl burn::module::ModuleDisplayDefault for Size {
    fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> {
        None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SizeRecord(pub Option<(usize, usize)>);

impl<B: Backend> burn::record::Record<B> for SizeRecord {
    type Item<S: burn::record::PrecisionSettings> = SizeRecord;

    fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: burn::record::PrecisionSettings>(
        item: Self::Item<S>,
        _device: &B::Device,
    ) -> Self {
        item
    }
}

impl From<(usize, usize)> for Size {
    fn from((h, w): (usize, usize)) -> Self {
        Self(h, w)
    }
}
impl<B: Backend> SamPredictor<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    /// Uses SAM to calculate the image embedding for an image, and then
    /// allow repeated, efficient mask prediction given prompts.
    /// Arguments:
    ///   sam_model (Sam): The model to use for mask prediction.
    pub fn new(sam_model: Sam<B>) -> Self {
        let target_length = sam_model.image_encoder.img_size;
        SamPredictor {
            model: sam_model,
            transfrom: ResizeLongestSide::new(target_length),
            is_image_set: false,
            features: None,
            orig_h: None,
            orig_w: None,
            input_size: None,
            original_size: None,
            input_h: None,
            input_w: None,
        }
    }

    /// Calculates the image embeddings for the provided image, allowing
    ///     masks to be predicted with the 'predict' method.
    ///     Arguments:
    ///       image (np.ndarray): The image for calculating masks. Expects an
    ///         image in HWC uint8 format, with pixel values in [0, 255].
    ///       image_format (str): The color format of the image, in ['RGB', 'BGR'].
    pub fn set_image(&mut self, image: Tensor<B, 3, Int>, image_format: ImageFormat) {
        // assert!(
        //     image.kind() == ,
        //     "Image should be uint8, but is {:?}",
        //     image.kind()
        // );

        let image = if image_format != self.model.image_format {
            image.flip([2]) // idk
        } else {
            image.clone()
        };

        let input_image = self.transfrom.apply_image(image.clone());
        let input_image = input_image.permute([2, 0, 1]).unsqueeze();
        let shape = image.dims();
        self.set_torch_image(input_image, Size(shape[0], shape[1]));
    }

    /// Calculates the image embeddings for the provided image, allowing
    /// masks to be predicted with the 'predict' method. Expects the input
    /// image to be already transformed to the format expected by the model.
    /// Arguments:
    ///   transformed_image (torch.Tensor): The input image, with shape
    ///     1x3xHxW, which has been transformed with ResizeLongestSide.
    ///   original_image_size (tuple(int, int)): The size of the image
    ///     before transformation, in (H, W) format.
    pub fn set_torch_image(&mut self, transformed_image: Tensor<B, 4, Int>, original_size: Size) {
        let shape = transformed_image.dims();
        assert!(
            shape.len() == 4
                && shape[1] == 3
                && shape[2..].iter().max().unwrap() == &self.model.image_encoder.img_size,
            "set_torch_image input must be BCHW with long side {}.",
            self.model.image_encoder.img_size,
        );
        self.original_size = Some(original_size);
        self.input_size = Some(Size(shape[2], shape[3]));
        let input_image = self.model.preprocess(transformed_image);
        let features = self.model.image_encoder.forward(input_image);
        
        #[cfg(test)]
        {
            let features_shape = features.shape();
            let features_data = features.clone().to_data();
            let features_vec: Vec<f32> = features_data.to_vec().unwrap();
            let mean: f32 = features_vec.iter().sum::<f32>() / features_vec.len() as f32;
            let variance: f32 = features_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / features_vec.len() as f32;
            let std = variance.sqrt();
            println!("Rust features shape: {:?}", features_shape);
            println!("Rust features mean: {}, std: {}", mean, std);
        }
        
        self.features = Some(features);
        self.is_image_set = true;
    }

    /// Predict masks for the given input prompts, using the currently set image.
    /// Arguments:
    ///   point_coords (np.ndarray or None): A Nx2 array of point prompts to the
    ///     model. Each point is in (X,Y) in pixels.
    ///   point_labels (np.ndarray or None): A length N array of labels for the
    ///     point prompts. 1 indicates a foreground point and 0 indicates a
    ///     background point.
    ///   box (np.ndarray or None): A length 4 array given a box prompt to the
    ///     model, in XYXY format.
    ///   mask_input (np.ndarray): A low resolution mask input to the model, typically
    ///     coming from a previous prediction iteration. Has form 1xHxW, where
    ///     for SAM, H=W=256.
    ///   multimask_output (bool): If true, the model will return three masks.
    ///     For ambiguous input prompts (such as a single click), this will often
    ///     produce better masks than a single prediction. If only a single
    ///     mask is needed, the model's predicted quality score can be used
    ///     to select the best mask. For non-ambiguous prompts, such as multiple
    ///     input prompts, multimask_output=False can give better results.
    ///   return_logits (bool): If true, returns un-thresholded masks logits
    ///     instead of a binary mask.
    /// Returns:
    ///   (np.ndarray): The output masks in CxHxW format, where C is the
    ///     number of masks, and (H, W) is the original image size.
    ///   (np.ndarray): An array of length C containing the model's
    ///     predictions for the quality of each mask.
    ///   (np.ndarray): An array of shape CxHxW, where C is the number
    ///     of masks and H=W=256. These low resolution logits can be passed to
    ///     a subsequent iteration as mask input.
    pub fn predict(
        &self,
        point_coords: Option<Tensor<B, 2, Int>>,
        point_labels: Option<Tensor<B, 1, Int>>,
        boxes: Option<Tensor<B, 2, Int>>,
        mask_input: Option<Tensor<B, 3>>,
        multimask_output: bool,
    ) -> (Tensor<B, 3, Bool>, Tensor<B, 1>, Tensor<B, 3>, Tensor<B, 3>) {
        // assert_eq!(
        //     &point_labels.unwrap().kind(),
        //     &Kind::Int,
        //     "point_labels must be int."
        // );
        assert!(self.is_image_set, "Must set image before predicting.");
        let (mut coords_torch, mut labels_torch, mut box_torch, mut mask_input_torch) =
            (None, None, None, None);
        if let Some(point_coords) = point_coords {
            let point_coords_float = point_coords.to_float();
            
            #[cfg(test)]
            {
                println!("SamPredictor.predict: original coords = {:?}", point_coords.clone().to_data().to_vec::<i64>().unwrap());
                println!("SamPredictor.predict: original_size = {:?}", self.original_size);
            }
            
            let point_coords = self
                .transfrom
                .apply_coords(point_coords_float, self.original_size.unwrap());
            
            #[cfg(test)]
            {
                println!("SamPredictor.predict: transformed coords = {:?}", point_coords.clone().to_data().to_vec::<f32>().unwrap());
            }
            
            coords_torch = Some(point_coords.unsqueeze());
            labels_torch = Some(
                point_labels
                    .expect("point_labels must be supplied if point_coords is supplied.")
                    .to_float()
                    .unsqueeze(),
            );
        }
        if let Some(boxes) = boxes {
            let boxes_float = boxes.to_float();
            let boxes = self
                .transfrom
                .apply_boxes(boxes_float, self.original_size.unwrap());
            box_torch = Some(boxes);
        }
        if let Some(mask_input) = mask_input {
            mask_input_torch = Some(mask_input.unsqueeze());
        }
        let (masks, iou_predictions, low_res_masks, mask_values) = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
        );
        let mask_values = mask_values.narrow(0, 0, 1).squeeze(0);
        let masks = masks.narrow(0, 0, 1).squeeze(0);
        let iou_predictions = iou_predictions.narrow(0, 0, 1).squeeze(0);
        let low_res_masks = low_res_masks.narrow(0, 0, 1).squeeze(0);
        (masks, iou_predictions, low_res_masks, mask_values)
    }
    /// Predict masks for the given input prompts, using the currently set image.
    /// Input prompts are batched torch tensors and are expected to already be
    /// transformed to the input frame using ResizeLongestSide.

    /// Arguments:
    ///   point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
    ///     model. Each point is in (X,Y) in pixels.
    ///   point_labels (torch.Tensor or None): A BxN array of labels for the
    ///     point prompts. 1 indicates a foreground point and 0 indicates a
    ///     background point.
    ///   boxes (np.ndarray or None): A Bx4 array given a box prompt to the
    ///     model, in XYXY format.
    ///   mask_input (np.ndarray): A low resolution mask input to the model, typically
    ///     coming from a previous prediction iteration. Has form Bx1xHxW, where
    ///     for SAM, H=W=256. Masks returned by a previous iteration of the
    ///     predict method do not need further transformation.
    ///   multimask_output (bool): If true, the model will return three masks.
    ///     For ambiguous input prompts (such as a single click), this will often
    ///     produce better masks than a single prediction. If only a single
    ///     mask is needed, the model's predicted quality score can be used
    ///     to select the best mask. For non-ambiguous prompts, such as multiple
    ///     input prompts, multimask_output=False can give better results.
    ///   return_logits (bool): If true, returns un-thresholded masks logits
    ///     instead of a binary mask.

    /// Returns:
    ///   (torch.Tensor): The output masks in BxCxHxW format, where C is the
    ///     number of masks, and (H, W) is the original image size.
    ///   (torch.Tensor): An array of shape BxC containing the model's
    ///     predictions for the quality of each mask.
    ///   (torch.Tensor): An array of shape BxCxHxW, where C is the number
    ///     of masks and H=W=256. These low res logits can be passed to
    ///     a subsequent iteration as mask input.
    pub fn predict_torch(
        &self,
        point_coords: Option<Tensor<B, 3>>,
        point_labels: Option<Tensor<B, 2>>,
        boxes: Option<Tensor<B, 2>>,
        mask_input: Option<Tensor<B, 4>>,
        multimask_output: bool,
    ) -> (Tensor<B, 4, Bool>, Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 4>) {
        assert!(self.is_image_set, "Must set image before predicting.");
        let point = match point_coords {
            Some(point_coords) => Some((point_coords, point_labels.unwrap())),
            None => None,
        };
        let (sparse_embeddings, dense_embeddings) =
            self.model
                .prompt_encoder
                .forward(point, boxes, mask_input, &Default::default());

        let (low_res_masks, iou_predictions) = self.model.mask_decoder.forward(
            self.features.clone().unwrap(),
            self.model.prompt_encoder.get_dense_pe(),
            sparse_embeddings,
            dense_embeddings,
            multimask_output,
        );
        let mask_values = self.model.postprocess_masks(
            low_res_masks.clone(),
            self.input_size.unwrap(),
            self.original_size.unwrap(),
        );
        let masks = mask_values.clone().greater_elem(self.model.mask_threshold);
        return (masks, iou_predictions, low_res_masks, mask_values);
    }
    /// Returns the image embeddings for the currently set image, with
    /// shape 1xCxHxW, where C is the embedding dimension and (H,W) are
    /// the embedding spatial dimension of SAM (typically C=256, H=W=64).
    pub fn get_image_embedding(&self) -> Tensor<B, 4> {
        assert!(self.is_image_set, "Must set image before predicting.");
        return self.features.clone().unwrap();
    }

    /// Resets the currently set image.
    pub fn reset_image(&mut self) {
        self.is_image_set = false;
        self.features = None;
        self.original_size = None;
        self.input_size = None;
        self.orig_h = None;
        self.orig_w = None;
        self.input_h = None;
        self.input_w = None;
    }
}

#[cfg(test)]
mod test {

    use burn::tensor::{Int, Tensor};
    use pyo3::types::PyAnyMethods;
    use pyo3::Bound;
    use pyo3::{types::PyTuple, PyAny, PyResult, Python};

    use crate::{
        python::python_data::{random_python_tensor, random_python_tensor_int, PythonData},
        tests::helpers::{get_python_test_sam, get_test_sam, TestBackend},
    };

    use super::{SamPredictor, Size};
    fn init(image: Option<Tensor<TestBackend, 3, Int>>) -> SamPredictor<TestBackend> {
        let device = Default::default();
        let sam = get_test_sam(&device);
        let mut predictor = SamPredictor::new(sam);
        if let Some(image) = image {
            predictor.set_image(image, super::ImageFormat::RGB);
        }

        predictor
    }
    fn python_init<'a>(
        py: &'a Python<'a>,
        with_set_image: bool,
    ) -> PyResult<(Bound<'a, PyAny>, Option<PythonData<3, i64>>)> {
        let sam = get_python_test_sam(py)?;
        let predictor = py
            .import("segment_anything.predictor")?
            .getattr("SamPredictor")?
            .call1((sam,))?;
        if with_set_image {
            let uint8 = py.import("torch")?.getattr("uint8")?;
            let image = random_python_tensor(*py, [120, 180, 3])?
                .call_method1("type", (uint8,))?
                .call_method0("numpy")?;
            predictor.call_method1("set_image", (&image, "RGB"))?;
            return Ok((predictor, Some(image.try_into()?)));
        }
        Ok((predictor, None))
    }

    #[test]
    fn test_predictor_set_image() {
        let python: PyResult<(PythonData<3, i64>, Size, Size, PythonData<4>)> =
            Python::attach(|py| {
                let (predictor, image) = python_init(&py, true)?;
                Ok((
                    image.unwrap().try_into()?,
                    predictor.getattr("original_size")?.try_into()?,
                    predictor.getattr("input_size")?.try_into()?,
                    predictor.getattr("features")?.try_into()?,
                ))
            });
        let (image, original_size, input_size, features) = python.unwrap();
        let predictor = init(Some(image.into()));
        assert_eq!(original_size, predictor.original_size.unwrap());
        assert_eq!(input_size, predictor.input_size.unwrap());
        features.almost_equal(predictor.features.unwrap(), 5.);
        assert!(predictor.is_image_set);
    }

    #[test]
    fn test_predictor_set_torch_image() {
        let original_size = (120, 180);
        let python: PyResult<(PythonData<4, i64>, Size, Size, PythonData<4>)> =
            Python::attach(|py| {
                let (predictor, _) = python_init(&py, false)?;
                let image = random_python_tensor_int(py, [1, 3, 683, 1024])?;
                predictor.call_method1("set_torch_image", (&image, original_size))?;
                Ok((
                    image.try_into()?,
                    predictor.getattr("original_size")?.try_into()?,
                    predictor.getattr("input_size")?.try_into()?,
                    predictor.getattr("features")?.try_into()?,
                ))
            });
        let (image, original_size, input_size, features) = python.unwrap();
        let mut predictor = init(None);

        predictor.set_torch_image(image.into(), original_size.into());
        assert_eq!(original_size, predictor.original_size.unwrap());
        assert_eq!(input_size, predictor.input_size.unwrap());
        features.almost_equal(predictor.features.unwrap(), 5.);
        assert!(predictor.is_image_set);
    }

    #[test]
    fn test_predictor_predict() {
        let python: PyResult<(
            PythonData<3, i64>,
            PythonData<2>,
            PythonData<1, i64>,
            PythonData<3>,
            PythonData<1>,
            PythonData<3>,
            PythonData<3>,
        )> = Python::attach(|py| {
            let (predictor, image) = python_init(&py, true)?;
            let point_coords = random_python_tensor(py, [1, 2])?.call_method0("numpy")?;
            let point_labels = random_python_tensor_int(py, [1])?.call_method0("numpy")?;
            let result = predictor.call_method1(
                "predict",
                (
                    &point_coords,
                    &point_labels,
                    py.None(),
                    py.None(),
                    true,
                    false,
                ),
            )?;
            let output = result.downcast::<PyTuple>()?;
            let masks = output.get_item(0)?;
            let iou_predictions = output.get_item(1)?;
            let low_res_masks = output.get_item(2)?;
            let mask_values = output.get_item(3)?;
            Ok((
                image.unwrap().try_into()?,
                point_coords.try_into()?,
                point_labels.try_into()?,
                masks.try_into()?,
                iou_predictions.try_into()?,
                low_res_masks.try_into()?,
                mask_values.try_into()?,
            ))
        });
        let (
            image,
            point_coords,
            point_labels,
            _masks,
            iou_predictions,
            low_res_masks,
            mask_values,
        ) = python.unwrap();
        let predictor = init(Some(image.into()));

        let (_masks2, iou_predictions2, low_res_masks2, mask_values2) = predictor.predict(
            Some(point_coords.into()),
            Some(point_labels.into()),
            None,
            None,
            true,
        );
        // masks.almost_equal(masks2, None);
        iou_predictions.almost_equal(iou_predictions2, 5.);
        low_res_masks.almost_equal(low_res_masks2, 5.);
        mask_values.almost_equal(mask_values2, 5.);
    }

    #[test]
    fn test_predictor_predict_torch() {
        let python: PyResult<(
            PythonData<3, i64>,
            PythonData<3>,
            PythonData<2, i64>,
            PythonData<4>,
            PythonData<2>,
            PythonData<4>,
            PythonData<4>,
        )> = Python::attach(|py| {
            let (predictor, image) = python_init(&py, true)?;
            let point_coords = random_python_tensor_int(py, [1, 1, 2])?;
            let point_labels = random_python_tensor_int(py, [1, 1])?;
            let result = predictor.call_method1(
                "predict_torch",
                (
                    &point_coords,
                    &point_labels,
                    py.None(),
                    py.None(),
                    true,
                    false,
                ),
            )?;
            let output = result.downcast::<PyTuple>()?;
            let masks = output.get_item(0)?;
            let iou_predictions = output.get_item(1)?;
            let low_res_masks = output.get_item(2)?;
            let mask_values = output.get_item(3)?;
            Ok((
                image.unwrap().try_into()?,
                point_coords.try_into()?,
                point_labels.try_into()?,
                masks.try_into()?,
                iou_predictions.try_into()?,
                low_res_masks.try_into()?,
                mask_values.try_into()?,
            ))
        });
        let (
            image,
            point_coords,
            point_labels,
            _masks,
            iou_predictions,
            low_res_masks,
            mask_values,
        ) = python.unwrap();
        let predictor = init(Some(image.into()));

        let (_masks2, iou_predictions2, low_res_masks2, mask_values2) = predictor.predict_torch(
            Some(point_coords.into()),
            Some(point_labels.into()),
            None,
            None,
            true,
        );
        // masks.almost_equal(masks2, None);
        iou_predictions.almost_equal(iou_predictions2, 5.);
        low_res_masks.almost_equal(low_res_masks2, 5.);
        mask_values.almost_equal(mask_values2, 5.);
    }
}
