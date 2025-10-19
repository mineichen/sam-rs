use std::{collections::HashMap, path::Path};

use crate::{build_sam::SamVersion, sam::Sam, tests::helpers::get_python_sam};
use burn::tensor::backend::Backend;
use pyo3::{
    types::{PyAnyMethods, PyTuple},
    Bound, PyAny, PyResult, Python,
};
use regex::Regex;

use super::update_tensor::update_tensor;

pub fn load_module_from_python<B: Backend>(
    sam: Sam<B>,
    version: SamVersion,
    file: &Path,
) -> PyResult<Sam<B>> {
    Python::attach(|py| {
        // if test then won't load the weights.
        let python_sam = get_python_sam(
            &py,
            version,
            match version {
                SamVersion::Test => None,
                _ => Some(file),
            },
        )?;

        // Saves the module to pth, in test version.
        // if version == SamVersion::Test {
        //     py.import("torch")?
        //         .call_method1("save", (python_sam, format!("{file}.pth")))?;
        // }

        let map = get_python_map(python_sam).unwrap();
        //let map: std::collections::HashMap<String, Bound<PyAny>> = std::collections::HashMap::new();
        println!("Python done");

        println!("Loading module in rust...");

        Ok(load_sam(sam.clone(), map))
    })
}

fn key_replacer(key: String) -> String {
    // Replacing "neck.1.", "output_upscaling.1.", "mask_downscaling.1." with neck1.
    let re = Regex::new(r"(neck|output_upscaling|mask_downscaling)\.(\d+)\.").unwrap();
    let key = re.replace_all(&key, "$1$2.").to_string();

    // Replacing all norm1.weight, norm2.weight with norm1.gamma, norm2.gamma
    let re = Regex::new(r"norm(\d+)\.weight").unwrap();
    let key = re.replace_all(&key, "norm$1.gamma").to_string();

    // Replacing all norm1.bias with norm1.beta
    let re = Regex::new(r"norm(\d+)\.bias").unwrap();
    let key = re.replace_all(&key, "norm$1.beta").to_string();

    // Replacing all norm_final_attn.weight with norm_final_attn.gamma
    let re = Regex::new(r"norm_final_attn\.weight").unwrap();
    let key = re.replace_all(&key, "norm_final_attn.gamma").to_string();

    // Replacing all norm_final_attn.bias with norm_final_attn.beta
    let re = Regex::new(r"norm_final_attn\.bias").unwrap();
    let key = re.replace_all(&key, "norm_final_attn.beta").to_string();

    // Replacing all .1. with [1].
    let re = Regex::new(r"\.(\d+)\.").unwrap();
    let key = re.replace_all(&key, "[$1].").to_string();

    key
}
pub fn get_python_map<'a>(sam: Bound<'a, PyAny>) -> PyResult<HashMap<String, Bound<'a, PyAny>>> {
    let mut map = HashMap::new();
    // Use state_dict() instead of named_parameters() to include buffers
    let state_dict = sam.call_method0("state_dict")?;
    let items = state_dict.call_method0("items")?;
    let items = items.try_iter()?;
    for item in items {
        let item_bound = item?;
        let item_tuple = item_bound.downcast::<PyTuple>()?;
        let key = item_tuple.get_item(0)?.extract::<String>()?;

        let key = key_replacer(key);

        let value = item_tuple.get_item(1)?;
        map.insert(key, value);
    }

    Ok(map)
}

// This function cannot be implemented because Sam doesn't implement Module trait
// The Module trait requires all fields to implement Module, but Sam contains
// non-Module types like [f32; 3], f32, and ImageFormat
#[allow(dead_code)]
pub fn load_sam<B: Backend>(mut sam: Sam<B>, values: HashMap<String, Bound<PyAny>>) -> Sam<B> {
    for (key, value) in values.into_iter() {
        update_tensor(&mut sam, &key, value.into_any());
    }
    // let record = sam.clone().into_record();
    // sam.load_record(record)
    sam
}
