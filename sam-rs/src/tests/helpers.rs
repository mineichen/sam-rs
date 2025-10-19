use std::path::Path;

use burn::{
    module::Module,
    record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder},
    tensor::backend::Backend,
};
#[cfg(feature = "pyo3")]
use pyo3::prelude::PyAnyMethods;
#[cfg(feature = "pyo3")]
use pyo3::{PyAny, PyResult, Python};

use crate::{build_sam::SamVersion, sam::Sam};

pub const TEST_ALMOST_THRESHOLD: f32 = 0.01;
// Using NdArray backend for tests (CPU-based, doesn't require GPU)
#[cfg(test)]
pub type TestBackend = burn_ndarray::NdArray<f32>;
pub const TEST_CHECKPOINT: &str = "../sam-convert/sam_test";
pub const TEST_SAM: SamVersion = SamVersion::Test;

pub fn get_sam<B: Backend>(
    version: SamVersion,
    checkpoint: Option<&Path>,
    device: &B::Device,
) -> Sam<B>
where
    <B as burn::tensor::backend::Backend>::FloatElem: From<f32>,
{
    let sam = version.build::<B>(checkpoint, &device);
    sam
}
#[cfg(test)]
pub fn get_test_sam(device: &<TestBackend as Backend>::Device) -> Sam<TestBackend> {
    // Don't load checkpoint for tests - use random weights
    get_sam(TEST_SAM, None, device)
}

#[cfg(feature = "pyo3")]
pub fn get_python_sam<'a>(
    py: &'a Python,
    version: SamVersion,
    checkpoint: Option<&Path>,
) -> PyResult<pyo3::Bound<'a, PyAny>> {
    let mut module = py
        .import("segment_anything.build_sam")?
        .getattr("sam_model_registry")?
        .get_item(version.to_str())?;

    match checkpoint {
        Some(checkpoint) => {
            let name = checkpoint.display().to_string();
            module = module.call1((name,))?;
        }
        None => module = module.call0()?,
    }

    Ok(module)
}

#[cfg(feature = "pyo3")]
pub fn get_python_test_sam<'a>(py: &'a Python) -> PyResult<pyo3::Bound<'a, PyAny>> {
    get_python_sam(&py, TEST_SAM, None)
}

pub fn load_module<B: Backend, D: Module<B>>(name: &str, module: D) -> D {
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::default();
    let path = format!(
        "{}/Documents/sam-models/{}.json",
        std::env::var("HOME").unwrap_or_else(|_| ".".to_string()),
        name
    );

    match recorder.load(path.clone().into(), &module.devices()[0]) {
        Ok(record) => module.load_record(record),
        Err(e) => {
            // Check if file exists but is in wrong format
            if std::path::Path::new(&path).exists() {
                eprintln!("\n⚠️  WARNING: Failed to load weights from {}", path);
                eprintln!("    Error: {:?}", e);
                eprintln!("    This file is likely in an old Burn format (0.8.0).");
                eprintln!(
                    "    Delete {} and re-run the test to regenerate in Burn 0.18.0 format.\n",
                    path
                );
                panic!("Weight file exists but is in incompatible format. Please delete it and re-run.");
            } else {
                // File doesn't exist - this is expected for first run
                eprintln!("\n📝 Note: No weight file found at {}", path);
                eprintln!(
                    "    Test will use random weights. To save weights for faster future runs,"
                );
                eprintln!("    update the test to call save_module() after loading weights from Python.\n");
                module
            }
        }
    }
}

pub fn save_module<B: Backend, M: Module<B>>(name: &str, module: &M) {
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::default();
    let path = format!(
        "{}/Documents/sam-models/{}.json",
        std::env::var("HOME").unwrap_or_else(|_| ".".to_string()),
        name
    );

    // Create directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    recorder
        .record(module.clone().into_record(), path.into())
        .unwrap_or_else(|e| {
            panic!("Failed to save module to {}: {:?}", name, e);
        });
}

/// Helper to check if a weight file exists in the current Burn format
pub fn weight_file_exists(name: &str) -> bool {
    std::path::Path::new(&format!(
        "{}/Documents/sam-models/{}.json",
        std::env::var("HOME").unwrap_or_else(|_| ".".to_string()),
        name
    ))
    .exists()
}
