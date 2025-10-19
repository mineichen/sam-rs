//! Test helpers specific to TinyViT modules

#[cfg(test)]
use pyo3::{
    types::{PyAnyMethods, PyDict},
    Bound, PyAny, PyResult, Python,
};

/// Helper to create Conv2d without bias for BatchNorm compatibility
#[cfg(test)]
pub fn create_conv2d_no_bias<'py>(
    py: Python<'py>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let torch_nn = py.import("torch.nn")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("bias", false)?;
    if groups > 1 {
        kwargs.set_item("groups", groups)?;
    }
    torch_nn.getattr("Conv2d")?.call(
        (in_channels, out_channels, kernel_size, stride, padding),
        Some(&kwargs),
    )
}

/// Helper to create a Conv2d_BN pair in a ModuleDict structure
#[cfg(test)]
pub fn create_conv2d_bn_dict<'py>(
    py: Python<'py>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let torch_nn = py.import("torch.nn")?;
    let conv = create_conv2d_no_bias(
        py,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
    )?;
    let bn = torch_nn.getattr("BatchNorm2d")?.call1((out_channels,))?;

    let dict = torch_nn.getattr("ModuleDict")?.call0()?;
    dict.call_method1("__setitem__", ("c", conv))?;
    dict.call_method1("__setitem__", ("bn", bn))?;
    Ok(dict)
}

/// Helper to rename BatchNorm parameters in saved JSON (weight→gamma, bias→beta)
/// This is needed because PyTorch uses "weight" and "bias" for BatchNorm,
/// but Burn uses "gamma" and "beta".
#[cfg(test)]
pub fn fix_batchnorm_names(py: Python, file: &str, paths: &[&str]) -> PyResult<()> {
    let fixes = paths
        .iter()
        .map(|p| {
            let path_ref = if p.is_empty() {
                "data['item']".to_string()
            } else {
                format!("data['item']['{}']", p)
            };
            format!(
                "if 'bn' in {}:\n    if 'weight' in {}['bn']:\n        {}['bn']['gamma'] = {}['bn'].pop('weight')\n    if 'bias' in {}['bn']:\n        {}['bn']['beta'] = {}['bn'].pop('bias')",
                path_ref, path_ref, path_ref, path_ref, path_ref, path_ref, path_ref
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let code = format!(
        r#"
import json
import os
path = os.path.expanduser('~/Documents/sam-models/{}.json')
with open(path, 'r') as f:
    data = json.load(f)
{}
with open(path, 'w') as f:
    json.dump(data, f)
"#,
        file, fixes
    );
    py.run(&std::ffi::CString::new(code).unwrap(), None, None)?;
    Ok(())
}

/// Save a PyTorch ModuleDict to file and fix BatchNorm parameter names
///
/// This helper combines the common pattern of:
/// 1. Setting module to eval mode
/// 2. Saving to file with module_to_file
/// 3. Renaming BatchNorm parameters (weight→gamma, bias→beta)
#[cfg(test)]
pub fn save_module_with_bn_fix<'py>(
    py: Python<'py>,
    file: &str,
    module: &Bound<'py, PyAny>,
    bn_paths: &[&str],
) -> PyResult<()> {
    use crate::python::module_to_file::module_to_file;

    // Set to eval mode
    module.call_method0("eval")?;

    // Save to file
    module_to_file(file, py, module)?;

    // Fix BatchNorm parameter names
    fix_batchnorm_names(py, file, bn_paths)?;

    Ok(())
}
