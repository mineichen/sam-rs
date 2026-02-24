use burn::tensor::{backend::Backend, Int, Tensor};
use image::{imageops::FilterType, ImageBuffer};

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

/// Resizes images to the longest side 'target_length', as well as provides
///  methods for resizing coordinates and boxes. Provides methods for
///  transforming both numpy array and batched torch tensors.
pub struct ResizeLongestSide {
    target_length: usize,
}
impl ResizeLongestSide {
    pub fn new(target_length: usize) -> Self {
        Self { target_length }
    }
    fn resize<B: Backend>(image: Tensor<B, 3, Int>, target_size: Size) -> Tensor<B, 3, Int> {
        let Size(tar_h, tar_w) = target_size;
        let device = image.device();
        let (image_data, shape) = image.to_slice::<i32>();
        let image_data = image_data.iter().map(|x| *x as u8).collect::<Vec<u8>>();
        let (height, width) = (shape[0], shape[1]);
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, image_data).unwrap();
        let resized_img =
            image::imageops::resize(&img, tar_w as u32, tar_h as u32, FilterType::CatmullRom);
        let resized_data = resized_img.into_raw().into_iter().map(|x| x as i32);
        Tensor::collect_shaped(resized_data, [tar_h, tar_w, 3], &device)
    }
    // Expects a numpy array with shape HxWxC in uint8 format.
    pub fn apply_image<B: Backend>(&self, image: Tensor<B, 3, Int>) -> Tensor<B, 3, Int> {
        let shape = image.dims();
        let target_size = self.get_preprocess_shape(shape[0], shape[1], self.target_length);
        return Self::resize(image, target_size);
    }

    // Expects a numpy array of length 2 in the final dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords<B: Backend, const D: usize>(
        &self,
        coords: Tensor<B, D>,
        original_size: Size,
    ) -> Tensor<B, D> {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(old_h, old_w, self.target_length);
        let coords_0 = coords.clone().narrow(D - 1, 0, 1) * (new_w as f32 / old_w as f32);
        let coords_1 = coords.narrow(D - 1, 1, 1) * (new_h as f32 / old_h as f32);
        Tensor::cat(vec![coords_0, coords_1], D - 1)
    }

    // Expects a numpy array shape Bx4. Requires the original image size
    // in (H, W) format.
    pub fn apply_boxes<B: Backend>(
        &self,
        boxes: Tensor<B, 2>,
        original_size: Size,
    ) -> Tensor<B, 2> {
        let boxes = self.apply_coords(boxes.reshape_max([usize::MAX, 2, 2]), original_size);
        boxes.reshape_max([usize::MAX, 4])
    }
    // Expects batched images with shape BxCxHxW and float format. This
    // transformation may not exactly match apply_image. apply_image is
    // the transformation expected by the model.
    //  Expects an image in BCHW format. May not exactly match apply_image.
    pub fn apply_image_torch<B: Backend>(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = image.dims();
        let (h, w) = (shape[2], shape[3]);
        let Size(target_h, target_w) = self.get_preprocess_shape(h, w, self.target_length);

        // Implement bilinear interpolation manually
        self.bilinear_interpolate(image, target_h, target_w)
    }

    fn bilinear_interpolate<B: Backend>(
        &self,
        image: Tensor<B, 4>,
        target_h: usize,
        target_w: usize,
    ) -> Tensor<B, 4> {
        let shape = image.dims();
        let (batch, channels, src_h, src_w) = (shape[0], shape[1], shape[2], shape[3]);
        let device = image.device();

        // Create target coordinate grids
        // i_coords: [target_h], j_coords: [target_w]
        let i_coords: Tensor<B, 1> = Tensor::arange(0..target_h as i64, &device).float();
        let j_coords: Tensor<B, 1> = Tensor::arange(0..target_w as i64, &device).float();

        // Compute source coordinates using align_corners=False formula
        // src_y = ((i + 0.5) * src_h / target_h - 0.5).max(0)
        // src_x = ((j + 0.5) * src_w / target_w - 0.5).max(0)
        let src_y: Tensor<B, 2> = i_coords
            .add_scalar(0.5)
            .mul_scalar(src_h as f32 / target_h as f32)
            .sub_scalar(0.5)
            .clamp_min(0.0)
            .reshape([target_h, 1])
            .repeat_dim(1, target_w);

        let src_x: Tensor<B, 2> = j_coords
            .add_scalar(0.5)
            .mul_scalar(src_w as f32 / target_w as f32)
            .sub_scalar(0.5)
            .clamp_min(0.0)
            .reshape([1, target_w])
            .repeat_dim(0, target_h);

        // Compute integer coordinates and weights
        let y0: Tensor<B, 2> = src_y.clone().floor();
        let y1: Tensor<B, 2> = y0.clone().add_scalar(1.0).clamp_max((src_h - 1) as f32);
        let x0: Tensor<B, 2> = src_x.clone().floor();
        let x1: Tensor<B, 2> = x0.clone().add_scalar(1.0).clamp_max((src_w - 1) as f32);

        let wy: Tensor<B, 2> = src_y - y0.clone();
        let wx: Tensor<B, 2> = src_x - x0.clone();

        // Convert to integer indices for gathering
        let y0_idx: Tensor<B, 2, burn::tensor::Int> = y0.int();
        let y1_idx: Tensor<B, 2, burn::tensor::Int> = y1.int();
        let x0_idx: Tensor<B, 2, burn::tensor::Int> = x0.int();
        let x1_idx: Tensor<B, 2, burn::tensor::Int> = x1.int();

        // Reshape image to [B, C, H, W] -> flatten spatial for gather
        // We'll use select along spatial dimensions

        // For each (y, x) position, gather the values
        // v00 = image[b, c, y0, x0], etc.

        // Expand indices for batch and channel dimensions
        // Indices need to be [B, target_h, target_w] for spatial selection
        let y0_flat: Tensor<B, 1, burn::tensor::Int> = y0_idx.reshape([target_h * target_w]);
        let y1_flat: Tensor<B, 1, burn::tensor::Int> = y1_idx.reshape([target_h * target_w]);
        let x0_flat: Tensor<B, 1, burn::tensor::Int> = x0_idx.reshape([target_h * target_w]);
        let x1_flat: Tensor<B, 1, burn::tensor::Int> = x1_idx.reshape([target_h * target_w]);

        // Reshape image to [B, C, H*W] for gathering along the spatial dimension
        let image_flat = image.reshape([batch, channels, src_h * src_w]);

        // Compute linear indices: y * src_w + x
        let idx00: Tensor<B, 1, burn::tensor::Int> =
            y0_flat.clone() * src_w as i64 + x0_flat.clone();
        let idx01: Tensor<B, 1, burn::tensor::Int> =
            y0_flat.clone() * src_w as i64 + x1_flat.clone();
        let idx10: Tensor<B, 1, burn::tensor::Int> =
            y1_flat.clone() * src_w as i64 + x0_flat.clone();
        let idx11: Tensor<B, 1, burn::tensor::Int> = y1_flat * src_w as i64 + x1_flat;

        // Gather values: select from spatial dimension
        let v00: Tensor<B, 3> = image_flat.clone().select(2, idx00);
        let v01: Tensor<B, 3> = image_flat.clone().select(2, idx01);
        let v10: Tensor<B, 3> = image_flat.clone().select(2, idx10);
        let v11: Tensor<B, 3> = image_flat.select(2, idx11);

        // Reshape weights to [1, 1, target_h * target_w] for broadcasting
        let wx_flat: Tensor<B, 1> = wx.reshape([target_h * target_w]);
        let wy_flat: Tensor<B, 1> = wy.reshape([target_h * target_w]);

        let one_minus_wx: Tensor<B, 1> = wx_flat.clone().neg().add_scalar(1.0);
        let one_minus_wy: Tensor<B, 1> = wy_flat.clone().neg().add_scalar(1.0);

        // Bilinear interpolation: v00 * (1-wx) * (1-wy) + v01 * wx * (1-wy) + v10 * (1-wx) * wy + v11 * wx * wy
        let w00 = one_minus_wx.clone() * one_minus_wy.clone();
        let w01 = wx_flat.clone() * one_minus_wy;
        let w10 = one_minus_wx * wy_flat.clone();
        let w11 = wx_flat * wy_flat;

        // Reshape weights for broadcasting over batch and channel
        let w00 = w00.reshape([1, 1, target_h * target_w]);
        let w01 = w01.reshape([1, 1, target_h * target_w]);
        let w10 = w10.reshape([1, 1, target_h * target_w]);
        let w11 = w11.reshape([1, 1, target_h * target_w]);

        let result: Tensor<B, 3> = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
        result.reshape([batch, channels, target_h, target_w])
    }

    // Expects a torch tensor with length 2 in the last dimension. Requires the
    // original image size in (H, W) format.
    pub fn apply_coords_torch<B: Backend, const D: usize>(
        &self,
        coords: Tensor<B, D>, //Expected to be Float
        original_size: Size,
    ) -> Tensor<B, D> {
        let Size(old_h, old_w) = original_size;
        let Size(new_h, new_w) = self.get_preprocess_shape(old_h, old_w, self.target_length);

        // Scale the coordinates - coords[..., 0] and coords[..., 1] in Python
        // This modifies only the first two elements of the last dimension
        let last_dim_size = coords.dims()[D - 1];

        // Scale column 0 and column 1
        let coords_0 = coords.clone().narrow(D - 1, 0, 1) * (new_w as f32 / old_w as f32);
        let coords_1 = coords.clone().narrow(D - 1, 1, 1) * (new_h as f32 / old_h as f32);

        // If there are more than 2 columns, keep the rest unchanged
        if last_dim_size > 2 {
            let coords_rest = coords.narrow(D - 1, 2, last_dim_size - 2);
            Tensor::cat(vec![coords_0, coords_1, coords_rest], D - 1)
        } else {
            Tensor::cat(vec![coords_0, coords_1], D - 1)
        }
    }

    // Expects a torch tensor with shape Bx4. Requires the original image
    // size in (H, W) format.
    pub fn apply_boxes_torch<B: Backend>(
        &self,
        boxes: Tensor<B, 2>,
        original_size: Size,
    ) -> Tensor<B, 2> {
        let boxes = self.apply_coords_torch(boxes.reshape_max([usize::MAX, 2, 2]), original_size);
        boxes.reshape_max([usize::MAX, 4])
    }

    // Compute the output size given input size and target long side length.
    pub fn get_preprocess_shape(&self, oldh: usize, oldw: usize, long_side_length: usize) -> Size {
        let scale = long_side_length as f32 / oldh.max(oldw) as f32;
        let newh = (oldh as f32 * scale) + 0.5;
        let neww = (oldw as f32 * scale) + 0.5;
        Size(newh as usize, neww as usize)
    }
}

#[cfg(test)]
mod test {
    use pyo3::{types::PyAnyMethods, Bound, PyAny, PyResult, Python};

    use crate::{
        python::python_data::{random_python_tensor, random_python_tensor_int, PythonData},
        sam_predictor::Size,
        tests::helpers::TestBackend,
    };
    //type TestBackend = burn_cpu::Cpu;

    fn python_module<'a>(py: &'a Python) -> PyResult<Bound<'a, PyAny>> {
        let module = py
            .import("segment_anything.utils.transforms")?
            .getattr("ResizeLongestSide")?;
        let module = module.call1((64,))?;
        Ok(module)
    }
    #[test]
    fn test_resize_get_preprocess_shape() {
        let python: PyResult<Size> = Python::attach(|py| {
            let module = python_module(&py)?;
            let output = module.call_method1("get_preprocess_shape", (32, 32, 64))?;
            Ok(output.try_into()?)
        });
        let python = python.unwrap();

        let resize = super::ResizeLongestSide::new(64);
        let output = resize.get_preprocess_shape(32, 32, 64);
        assert_eq!(python, output);
    }
    #[test]
    fn test_resize_apply_image() {
        let python: PyResult<(PythonData<3, i64>, PythonData<3, i64>)> = Python::attach(|py| {
            let module = python_module(&py)?;
            let uint8 = py.import("torch")?.getattr("uint8")?;
            let input = random_python_tensor(py, [120, 180, 3])?
                .call_method1("type", (uint8,))?
                .call_method0("numpy")?;

            let output = module.call_method1("apply_image", (input.clone(),))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize
            .apply_image::<TestBackend>(input.into_tensor::<TestBackend, _>(&Default::default()));
        python.almost_equal(output, Some(50.));
    }
    #[test]
    fn test_resize_apply_coords() {
        let original_size = (1200, 1800);
        let python: PyResult<(PythonData<3>, PythonData<3>)> = Python::attach(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [1, 2, 2])?
                .getattr("numpy")?
                .call0()?;
            let output = module.call_method1("apply_coords", (input.clone(), original_size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_coords::<TestBackend, 3>(
            input.into_tensor::<TestBackend, _>(&Default::default()),
            original_size.into(),
        );
        python.almost_equal(output, None);
    }

    #[test]
    fn test_resize_apply_boxes() {
        let original_size = (1200, 1800);
        let python: PyResult<(PythonData<2>, PythonData<2>)> = Python::attach(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [1, 4])?
                .getattr("numpy")?
                .call0()?;
            let output = module.call_method1("apply_boxes", (input.clone(), original_size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_boxes::<TestBackend>(
            input.into_tensor::<TestBackend, _>(&Default::default()),
            original_size.into(),
        );
        python.almost_equal(output, None);
    }

    #[test]
    fn test_resize_image_torch() {
        let python: PyResult<(PythonData<4>, PythonData<4>)> = Python::attach(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor(py, [1, 3, 32, 32])?;
            let output = module.call_method1("apply_image_torch", (input.clone(),))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_image_torch::<TestBackend>(
            input.into_tensor::<TestBackend, _>(&Default::default()),
        );
        python.almost_equal(output, None);
    }
    #[test]
    fn test_resize_coords_torch() {
        let size = (32, 32);
        let python: PyResult<(PythonData<2>, PythonData<2>)> = Python::attach(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [32, 32])?;
            let output = module.call_method1("apply_coords_torch", (input.clone(), size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_coords_torch::<TestBackend, 2>(
            input.into_tensor::<TestBackend, _>(&Default::default()),
            size.into(),
        );
        python.almost_equal(output, None);
    }
    #[test]
    fn test_resize_boxes_torch() {
        let size = (32, 32);
        let python: PyResult<(PythonData<2>, PythonData<2>)> = Python::attach(|py| {
            let module = python_module(&py)?;
            let input = random_python_tensor_int(py, [32, 32])?;
            let output = module.call_method1("apply_boxes_torch", (input.clone(), size))?;
            Ok((input.try_into()?, output.try_into()?))
        });
        let (input, python) = python.unwrap();
        let resize = super::ResizeLongestSide::new(64);
        let output = resize.apply_boxes_torch::<TestBackend>(
            input.into_tensor::<TestBackend, _>(&Default::default()),
            size.into(),
        );
        python.almost_equal(output, None);
    }
}
