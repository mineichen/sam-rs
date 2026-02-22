use std::path::Path;
use std::time::Instant;

use burn::tensor::Tensor;
use burn_ndarray::NdArray;

use sam_rs::build_sam::SamVersion;
use sam_rs::burn_helpers::TensorHelpers;
use sam_rs::helpers::load_image;
use sam_rs::sam_predictor::{ImageFormat, SamPredictor, Size};

type Backend = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).fuse();

    let image_path = args
        .next()
        .unwrap_or_else(|| "images/truck.jpg".to_string());
    let pos_x: i64 = args.next().and_then(|x| x.parse().ok()).unwrap_or(770);
    let pos_y: i64 = args.next().and_then(|x| x.parse().ok()).unwrap_or(377);
    let checkpoint = args
        .next()
        .unwrap_or_else(|| "sam-convert/sam_vit_b_01ec64".to_string());

    let mut last_time = Instant::now();
    let mut elapsed = || {
        let e = last_time.elapsed();
        last_time = Instant::now();
        e.as_secs_f64()
    };

    println!("Loading SAM model...");
    println!("Checkpoint: {}", checkpoint);
    let device = Default::default();
    let sam = SamVersion::VitB.build::<Backend>(Some(Path::new(&checkpoint)), &device);
    let mut predictor = SamPredictor::new(sam);
    println!("Model loaded [{:.3}s]", elapsed());

    println!("Loading image: {}", image_path);
    let (image, Size(orig_h, orig_w)) = load_image(&image_path, &device);
    println!("Image size: {}x{} [{:.3}s]", orig_w, orig_h, elapsed());

    println!("Setting image...");
    predictor.set_image(image, ImageFormat::RGB);
    println!("Image set [{:.3}s]", elapsed());

    println!("Running prediction for point ({}, {})...", pos_x, pos_y,);
    let point_coords = Tensor::of_slice(vec![pos_x, pos_y], [1, 2], &device);
    let point_labels = Tensor::of_slice(vec![1i64], [1], &device);

    let (masks, iou_predictions, _, _) =
        predictor.predict(Some(point_coords), Some(point_labels), None, None, true);

    let iou_data = iou_predictions.to_data();
    let iou_slice = iou_data.as_slice::<f32>().unwrap();
    println!("Prediction done [{:.3}s]", elapsed());
    println!("IoU predictions: {:?}", iou_slice);

    let best_idx = iou_slice
        .iter()
        .enumerate()
        .max_by(|(_, a): &(_, &f32), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    println!(
        "Best mask index: {} (IoU: {:.4}) [{:.3}s]",
        best_idx,
        iou_slice[best_idx],
        elapsed()
    );

    let best_mask = masks.narrow(0, best_idx, 1).squeeze::<2>();
    let mask_data = best_mask.to_data();
    let mask_slice = mask_data.as_slice::<bool>().unwrap();

    let original_img = image::open(&image_path)?.to_rgb8();
    let (width, height) = original_img.dimensions();

    let mut output_img = original_img.clone();
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if idx < mask_slice.len() && mask_slice[idx] {
                let pixel = output_img.get_pixel_mut(x, y);
                pixel[0] = pixel[0].saturating_add(100);
                pixel[1] = pixel[1].saturating_sub(50);
                pixel[2] = pixel[2].saturating_sub(50);
            }
        }
    }

    let output_path = format!(
        "{}_mask_{}.png",
        Path::new(&image_path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap(),
        best_idx
    );
    output_img.save(&output_path)?;
    println!("Saved mask to: {} [{:.3}s]", output_path, elapsed());

    let mask_path = format!(
        "{}_mask_raw.png",
        Path::new(&image_path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
    );
    let mut mask_img = image::GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if idx < mask_slice.len() {
                mask_img.put_pixel(
                    x,
                    y,
                    if mask_slice[idx] {
                        image::Luma([255u8])
                    } else {
                        image::Luma([0u8])
                    },
                );
            }
        }
    }
    mask_img.save(&mask_path)?;
    println!("Saved raw mask to: {} [{:.3}s]", mask_path, elapsed());

    Ok(())
}
