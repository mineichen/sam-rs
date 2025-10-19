use burn::tensor::{backend::Backend, Int, Tensor};
// use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};
use opencv::{
    core::{self, Vec3b},
    imgcodecs, imgproc,
    prelude::{Mat, MatTraitConst},
};

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

pub fn load_image<B: Backend>(image_path: &str, device: &B::Device) -> (Tensor<B, 3, Int>, Size) {
    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR).unwrap();
    // Convert BGR to RGB by swapping channels
    let mut rgb_image = Mat::default();
    imgproc::cvt_color(
        &image,
        &mut rgb_image,
        imgproc::COLOR_BGR2RGB,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .unwrap_or_else(|_| {
        // If color conversion fails, just use the original image
        rgb_image = image;
    });

    let size = rgb_image.size().unwrap();
    let size = Size(size.height as usize, size.width as usize);

    let mut slice = Vec::with_capacity(size.0 * size.1 * 3);

    for row in 0..size.0 {
        for col in 0..size.1 {
            let pixel: Vec3b = *rgb_image.at_2d(row as i32, col as i32).unwrap();
            for value in pixel {
                slice.push(value as i32);
            }
        }
    }
    let shape = [size.0, size.1, 3];
    let image = Tensor::of_slice(slice, shape, device);
    (image, size)
}

#[cfg(test)]
mod test {

    use pyo3::types::PyAnyMethods;
    use pyo3::{PyResult, Python};

    use crate::{python::python_data::PythonData, tests::helpers::TestBackend};

    use super::load_image;

    fn load_python_image(file: &str) -> PyResult<PythonData<3>> {
        Python::attach(|py| {
            let cv2 = py.import("cv2")?;
            let image = cv2.call_method1("imread", (file,))?;
            let image = cv2.call_method1("cvtColor", (image, cv2.getattr("COLOR_BGR2RGB")?))?;
            Ok(image.try_into()?)
        })
    }
    #[test]
    fn test_image_loading() {
        let file = "../images/truck.jpg";
        let python_image = load_python_image(file).unwrap();
        let device = Default::default();
        let (image, _) = load_image::<TestBackend>(file, &device);
        python_image.almost_equal(image, None);
    }
}
