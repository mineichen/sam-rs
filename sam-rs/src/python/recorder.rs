use std::{collections::HashMap, path::Path};

use crate::{build_sam::SamVersion, sam::Sam, tests::helpers::get_python_sam};
use burn::tensor::backend::Backend;
use pyo3::{
    types::{PyAnyMethods, PyTuple},
    Bound, PyAny, PyResult, Python,
};

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
        //         .call_method1("save", (python_sam, format!("{file}")))?;
        // }

        let map = get_python_map(python_sam).unwrap();
        //let map: std::collections::HashMap<String, Bound<PyAny>> = std::collections::HashMap::new();
        println!("Python done");

        println!("Loading module in rust...");

        Ok(load_sam(sam.clone(), map))
    })
}

fn key_replacer(mut key: String) -> String {
    // Replacing "neck.1.", "output_upscaling.1.", "mask_downscaling.1." with neck1.
    for prefix in ["neck", "output_upscaling", "mask_downscaling"] {
        let pattern = format!("{}.", prefix);
        if let Some(pos) = key.find(&pattern) {
            let after_prefix = pos + pattern.len();
            if after_prefix < key.len() {
                // Find the digit(s) after the prefix
                let rest = &key[after_prefix..];
                if let Some(digit_end) = rest.find(|c: char| !c.is_ascii_digit()) {
                    if digit_end > 0 && rest.chars().nth(digit_end) == Some('.') {
                        // Extract the digit(s)
                        let digits = &rest[..digit_end];
                        // Reconstruct: prefix + digits + rest after the dot
                        key = format!(
                            "{}{}.{}",
                            &key[..pos + prefix.len()],
                            digits,
                            &rest[digit_end + 1..]
                        );
                    }
                }
            }
        }
    }

    // Replacing all norm_final_attn.weight with norm_final_attn.gamma
    key = key.replace("norm_final_attn.weight", "norm_final_attn.gamma");

    // Replacing all norm_final_attn.bias with norm_final_attn.beta
    key = key.replace("norm_final_attn.bias", "norm_final_attn.beta");

    // Replacing all norm1.weight, norm2.weight with norm1.gamma, norm2.gamma
    // Look for "norm" followed by digits and ".weight"
    if let Some(norm_pos) = key.find("norm") {
        let after_norm = norm_pos + 4;
        if after_norm < key.len() {
            let rest = &key[after_norm..];
            if let Some(first_char) = rest.chars().next() {
                if first_char.is_ascii_digit() {
                    // Find where digits end
                    if let Some(digit_end) = rest.find(|c: char| !c.is_ascii_digit()) {
                        if rest[digit_end..].starts_with(".weight") {
                            let before_weight = after_norm + digit_end;
                            key = format!(
                                "{}.gamma{}",
                                &key[..before_weight],
                                &key[before_weight + 7..]
                            );
                        } else if rest[digit_end..].starts_with(".bias") {
                            let before_bias = after_norm + digit_end;
                            key =
                                format!("{}.beta{}", &key[..before_bias], &key[before_bias + 5..]);
                        }
                    }
                }
            }
        }
    }

    // Replacing all .1. with [1].
    let mut result = String::new();
    let chars: Vec<char> = key.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '.' && i + 2 < chars.len() {
            // Look ahead to see if we have .digit(s).
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }

            // If we found digits followed by a dot
            if j > i + 1 && j < chars.len() && chars[j] == '.' {
                // Extract the digits
                result.push('[');
                for k in (i + 1)..j {
                    result.push(chars[k]);
                }
                result.push(']');
                result.push('.');
                i = j + 1;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_replacer() {
        // Test replacing neck.1. with neck1.
        assert_eq!(key_replacer("neck.1.layer".to_string()), "neck1.layer");
        assert_eq!(
            key_replacer("output_upscaling.2.weight".to_string()),
            "output_upscaling2.weight"
        );
        assert_eq!(
            key_replacer("mask_downscaling.10.bias".to_string()),
            "mask_downscaling10.bias"
        );

        // Test replacing norm_final_attn.weight with norm_final_attn.gamma
        assert_eq!(
            key_replacer("norm_final_attn.weight".to_string()),
            "norm_final_attn.gamma"
        );
        assert_eq!(
            key_replacer("norm_final_attn.bias".to_string()),
            "norm_final_attn.beta"
        );

        // Test replacing norm1.weight with norm1.gamma
        assert_eq!(key_replacer("norm1.weight".to_string()), "norm1.gamma");
        assert_eq!(key_replacer("norm2.bias".to_string()), "norm2.beta");
        assert_eq!(key_replacer("norm10.weight".to_string()), "norm10.gamma");

        // Test replacing .1. with [1].
        assert_eq!(
            key_replacer("layer.1.weight".to_string()),
            "layer[1].weight"
        );
        assert_eq!(
            key_replacer("blocks.0.attn.1.proj".to_string()),
            "blocks[0].attn[1].proj"
        );
        assert_eq!(
            key_replacer("encoder.12.layer".to_string()),
            "encoder[12].layer"
        );

        // Test combinations
        assert_eq!(
            key_replacer("neck.1.layer.2.norm1.weight".to_string()),
            "neck1.layer[2].norm1.gamma"
        );
    }
}
