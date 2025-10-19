#[cfg(test)]
mod test {
    extern crate ndarray;

    use std::path::Path;

    use burn::tensor::Tensor;
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyTuple};
    use pyo3::{PyResult, Python};

    use crate::build_sam::SamVersion;
    use crate::burn_helpers::TensorHelpers;
    use crate::helpers::load_image;
    use crate::python::python_data::PythonData;
    use crate::sam_predictor::{ImageFormat, SamPredictor};
    use crate::tests::helpers::{get_python_sam, get_sam, TestBackend};

    #[test]
    #[ignore]
    fn test_prediction() {
        let image_path = "../images/dog.jpg";
        let version = SamVersion::VitB;
        let checkpoint = Some(Path::new("../sam-convert/sam_vit_b_01ec64"));
        let inputs = vec![170, 375];
        let labels = vec![1];

        // Remove generated images at start so we immediately see if they weren't regenerated
        let rust_output = get_prediction_output_path(image_path, "_rust");
        let python_output = get_prediction_output_path(image_path, "_python");
        let _ = std::fs::remove_file(&rust_output);
        let _ = std::fs::remove_file(&python_output);

        let python: PyResult<(
            PythonData<3>,
            PythonData<3>,
            PythonData<1>,
            PythonData<3>,
            PythonData<3>,
        )> = Python::attach(|py| {
            let cv2 = py.import("cv2")?;

            // Loading image
            let image = cv2.call_method1("imread", (image_path,))?;
            let image = cv2.call_method1("cvtColor", (image, cv2.getattr("COLOR_BGR2RGB")?))?;

            //Setting image
            let sam = get_python_sam(&py, version, checkpoint)?;
            println!("Python SAM model type: {:?}", version);
            println!("Python SAM checkpoint: {:?}", checkpoint);
            let predictor = py
                .import("segment_anything.predictor")?
                .call_method1("SamPredictor", (sam,))?;

            predictor.call_method1("set_image", (&image,))?;
            let np = py.import("numpy")?;
            let input_point = np.call_method1("array", (vec![inputs.clone()],))?;
            let input_label = np
                .call_method1("array", (labels.clone(),))?
                .call_method1("astype", (np.getattr("int64")?,))?;

            // Debug: Check what Python's predictor has for coordinates
            println!("\nPython predictor debug:");
            println!(
                "Python original_size: {:?}",
                predictor.getattr("original_size")?
            );
            println!("Python input_size: {:?}", predictor.getattr("input_size")?);

            // Transform coordinates the same way Python does to compare
            let transform = predictor.getattr("transform")?;
            let transformed_coords = transform.call_method1(
                "apply_coords",
                (input_point.clone(), predictor.getattr("original_size")?),
            )?;
            println!("Python transformed coords: {:?}", transformed_coords);

            // Check image features
            let features = predictor.getattr("features")?;
            let features_shape = features.getattr("shape")?;
            println!("Python features shape: {:?}", features_shape);
            // Get some feature values to compare
            let features_flat = features.call_method0("flatten")?;
            let features_mean = features_flat.call_method0("mean")?;
            let features_std = features_flat.call_method0("std")?;
            println!(
                "Python features mean: {:?}, std: {:?}",
                features_mean, features_std
            );

            //Predicting
            let kwargs = PyDict::new(py);
            kwargs.set_item("point_coords", input_point)?;
            kwargs.set_item("point_labels", input_label)?;
            kwargs.set_item("multimask_output", true)?;
            let output = predictor.call_method("predict", (), Some(&kwargs))?;
            let output = output.downcast::<PyTuple>()?;

            let masks = output.get_item(0)?;
            let scores = output.get_item(1)?;
            let logits = output.get_item(2)?;
            let mask_values = output.get_item(3)?;

            Ok((
                image.try_into()?,
                masks.try_into()?,
                scores.try_into()?,
                logits.try_into()?,
                mask_values.try_into()?,
            ))
        });
        let (image, masks, scores, logits, mask_values) = python.unwrap();
        let device = Default::default();
        println!(
            "Loading SAM model with version: {:?}, checkpoint: {:?}",
            version, checkpoint
        );
        let sam = get_sam::<TestBackend>(version, checkpoint, &device);
        println!("SAM model loaded successfully");

        // Debug: Check if mask decoder weights look reasonable (not random)
        println!("Checking loaded model parameters...");
        println!("Pixel mean: {:?}", sam.pixel_mean);
        println!("Pixel std: {:?}", sam.pixel_std);

        // Check if positional encoding matrix was loaded correctly
        let pe_matrix = sam
            .prompt_encoder
            .pe_layer
            .positional_encoding_gaussian_matrix
            .val();
        let pe_shape = pe_matrix.shape();
        let pe_data = pe_matrix.clone().to_data();
        let pe_vec: Vec<f32> = pe_data.to_vec().unwrap();
        println!("Positional encoding matrix shape: {:?}", pe_shape);
        println!(
            "Positional encoding matrix first 10 values: {:?}",
            &pe_vec[..10.min(pe_vec.len())]
        );
        println!(
            "Positional encoding matrix mean: {:.4}, std: {:.4}",
            pe_vec.iter().sum::<f32>() / pe_vec.len() as f32,
            {
                let mean = pe_vec.iter().sum::<f32>() / pe_vec.len() as f32;
                let variance =
                    pe_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / pe_vec.len() as f32;
                variance.sqrt()
            }
        );

        let mut predictor = SamPredictor::new(sam);

        // Loading image
        let (image2, _) = load_image(image_path, &device);

        // Setting image
        predictor.set_image(image2.clone(), ImageFormat::RGB);

        //Example inputs
        let input_point = Tensor::of_slice(inputs, [1, 2], &device);
        let input_label = Tensor::of_slice(labels, [1], &device);

        println!(
            "Input point: {:?}",
            input_point.clone().to_data().to_vec::<i64>().unwrap()
        );
        println!(
            "Input label: {:?}",
            input_label.clone().to_data().to_vec::<i64>().unwrap()
        );

        let (masks2, scores2, logits2, mask_values2) =
            predictor.predict(Some(input_point), Some(input_label), None, None, true);

        // Save both Python and Rust prediction outputs as PNGs
        save_prediction_image(
            image_path,
            &image2,
            &masks2,
            &scores2,
            &mask_values2,
            "_rust",
        );

        // Convert Python tensors to Rust tensors for visualization
        let python_masks_tensor: Tensor<TestBackend, 3> = masks.clone().into();
        let python_scores_tensor: Tensor<TestBackend, 1> = scores.clone().into();
        let python_mask_values_tensor: Tensor<TestBackend, 3> = mask_values.clone().into();

        println!("\nDebug Python tensor conversion:");
        println!(
            "Python mask_values shape: {:?}",
            python_mask_values_tensor.shape()
        );
        println!(
            "Python mask_values first 20: {:?}",
            &mask_values.slice[..20.min(mask_values.slice.len())]
        );
        println!("Image2 shape for comparison: {:?}", image2.shape());

        save_prediction_image(
            image_path,
            &image2,
            &python_masks_tensor,
            &python_scores_tensor,
            &python_mask_values_tensor,
            "_python",
        );

        // Debug: Compare predictions with Python baseline
        let python_scores_vec: Vec<f32> = scores.slice.iter().copied().collect();
        let rust_scores_vec: Vec<f32> = scores2.clone().to_data().to_vec().unwrap();
        println!("\n=== Comparison Results ===");
        println!("Python scores: {:?}", python_scores_vec);
        println!("Rust scores: {:?}", rust_scores_vec);

        println!("\nPython logits shape: {:?}", logits.shape);
        println!("Rust logits shape: {:?}", logits2.shape());
        println!(
            "Python logits first 10: {:?}",
            &logits.slice[..10.min(logits.slice.len())]
        );
        let rust_logits_vec: Vec<f32> = logits2.clone().to_data().to_vec().unwrap();
        println!(
            "Rust logits first 10: {:?}",
            &rust_logits_vec[..10.min(rust_logits_vec.len())]
        );

        println!("\nPython mask_values shape: {:?}", mask_values.shape);
        println!("Rust mask_values shape: {:?}", mask_values2.shape());
        println!(
            "Python mask_values first 10: {:?}",
            &mask_values.slice[..10.min(mask_values.slice.len())]
        );
        let rust_mask_values_vec: Vec<f32> = mask_values2.clone().to_data().to_vec().unwrap();
        println!(
            "Rust mask_values first 10: {:?}",
            &rust_mask_values_vec[..10.min(rust_mask_values_vec.len())]
        );

        println!("\nPython masks shape: {:?}", masks.shape);
        println!("Rust masks shape: {:?}", masks2.shape());
        println!("=== End Comparison ===\n");

        image.almost_equal(image2, 5.);
        scores.almost_equal(scores2, None);
        logits.almost_equal(logits2, None);
        mask_values.almost_equal(mask_values2, None);
        masks.almost_equal(masks2, None);
    }

    fn save_prediction_image<K: burn::tensor::TensorKind<TestBackend>>(
        original_path: &str,
        image: &Tensor<TestBackend, 3, burn::tensor::Int>,
        _masks: &Tensor<TestBackend, 3, K>,
        scores: &Tensor<TestBackend, 1>,
        mask_values: &Tensor<TestBackend, 3>, // High-res logits at image resolution
        postfix: &str,
    ) {
        use image::{ImageBuffer, Rgb};

        // Get image dimensions [H, W, C] - load_image returns in this format
        let image_shape = image.shape();
        let height = image_shape.dims[0];
        let width = image_shape.dims[1];

        println!(
            "save_prediction_image{}: image shape = {:?}",
            postfix, image_shape
        );

        // Image is already in [H, W, C] format from load_image
        let image_data = image.clone().to_data();
        let image_vec: Vec<i64> = image_data.to_vec().unwrap();

        // Get scores [N] to find best mask
        let scores_data = scores.clone().to_data();
        let scores_vec: Vec<f32> = scores_data.to_vec().unwrap();

        // Find best mask (highest score)
        let best_idx = scores_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        println!(
            "save_prediction_image{}: best_idx = {}, scores = {:?}",
            postfix, best_idx, scores_vec
        );

        // Get mask_values [N, H, W] for confidence per pixel at full resolution
        let mask_values_shape = mask_values.shape();
        let mask_values_h = mask_values_shape.dims[1];
        let mask_values_w = mask_values_shape.dims[2];

        println!(
            "save_prediction_image{}: mask_values shape = [{}, {}, {}]",
            postfix, mask_values_shape.dims[0], mask_values_h, mask_values_w
        );

        // Verify dimensions match
        assert_eq!(
            mask_values_h, height,
            "mask_values height must match image height"
        );
        assert_eq!(
            mask_values_w, width,
            "mask_values width must match image width"
        );

        let mask_values_data = mask_values.clone().to_data();
        let mask_values_vec: Vec<f32> = mask_values_data.to_vec().unwrap();

        let mask_size = height * width;
        let best_mask_values = &mask_values_vec[best_idx * mask_size..(best_idx + 1) * mask_size];

        // Sigmoid function to convert logits to probabilities [0, 1]
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

        // Create output image buffer
        let mut img_buffer = ImageBuffer::new(width as u32, height as u32);

        for i in 0..height {
            for j in 0..width {
                let pixel_idx = i * width + j;
                let mask_value = best_mask_values[pixel_idx];
                let confidence = sigmoid(mask_value); // 0.0 to 1.0

                let img_idx = (i * width + j) * 3;

                // Image is stored as RGB, blend each channel
                let r = (image_vec[img_idx] as f32 * confidence + 255.0 * (1.0 - confidence)) as u8;
                let g =
                    (image_vec[img_idx + 1] as f32 * confidence + 255.0 * (1.0 - confidence)) as u8;
                let b =
                    (image_vec[img_idx + 2] as f32 * confidence + 255.0 * (1.0 - confidence)) as u8;

                img_buffer.put_pixel(j as u32, i as u32, Rgb([r, g, b]));
            }
        }

        // Generate output filename with postfix and save
        let output_path = get_prediction_output_path(original_path, postfix);
        img_buffer.save(&output_path).unwrap();
        println!("Saved prediction to: {}", output_path.display());
    }
    fn get_prediction_output_path(original_path: &str, postfix: &str) -> std::path::PathBuf {
        let path = Path::new(original_path);
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir)
            .parent()
            .unwrap()
            .join("target")
            .join(format!("{}_predict{}.png", stem, postfix))
    }
}
