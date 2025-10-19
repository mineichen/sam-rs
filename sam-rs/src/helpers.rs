use burn::tensor::{backend::Backend, Int, Tensor};
// use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};
use image::ImageReader;

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

pub fn load_image<B: Backend>(image_path: &str, device: &B::Device) -> (Tensor<B, 3, Int>, Size) {
    let img = ImageReader::open(image_path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    let (width, height) = img.dimensions();
    let size = Size(height as usize, width as usize);

    let slice = img.into_vec();
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
        use tempfile::Builder;

        // Load JPEG with image crate and save to temporary PNG, as Opencv and image decoding jpg is not exactly the same
        let original_file = "../images/truck.jpg";
        let img = image::ImageReader::open(original_file)
            .unwrap()
            .decode()
            .unwrap();

        // Create temp file with .png extension
        let temp_file = Builder::new().suffix(".png").tempfile().unwrap();
        let temp_path = temp_file.path();
        img.save(temp_path).unwrap();

        // Now load the PNG with both Python and Rust - should be identical
        let python_image = load_python_image(temp_path.to_str().unwrap()).unwrap();
        let device = Default::default();
        let (image, _) = load_image::<TestBackend>(temp_path.to_str().unwrap(), &device);

        // PNG is lossless, so both decoders should produce identical results
        python_image.almost_equal(image, None);
    }
}
