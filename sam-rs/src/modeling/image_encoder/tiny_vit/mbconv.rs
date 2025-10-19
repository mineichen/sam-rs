use burn::{
    module::Module,
    nn::{DropoutConfig, Gelu},
    tensor::{backend::Backend, Tensor},
};

use super::conv2d_bn::Conv2dBN;

/// Mobile Inverted Bottleneck Convolution (MBConv)
///
/// An efficient convolutional block that:
/// 1. Expands channels with 1x1 conv
/// 2. Applies depthwise 3x3 conv
/// 3. Projects back with 1x1 conv
/// 4. Uses residual connection if in_chans == out_chans
#[derive(Module, Debug)]
pub struct MBConv<B: Backend> {
    in_chans: usize,
    out_chans: usize,
    // Expansion: 1x1 conv
    conv1: Conv2dBN<B>,
    act1: Gelu,
    // Depthwise: 3x3 conv with groups
    conv2: Conv2dBN<B>,
    act2: Gelu,
    // Projection: 1x1 conv
    conv3: Conv2dBN<B>,
    act3: Gelu,
    drop_path: burn::nn::Dropout,
}

impl<B: Backend> MBConv<B> {
    /// Create a new MBConv module
    ///
    /// # Arguments
    /// * `in_chans` - Number of input channels
    /// * `out_chans` - Number of output channels
    /// * `expand_ratio` - Channel expansion ratio (typically 4.0)
    /// * `drop_path` - Drop path rate for stochastic depth
    /// * `device` - Device to create the module on
    pub fn new(
        in_chans: usize,
        out_chans: usize,
        expand_ratio: f64,
        drop_path: f64,
        device: &B::Device,
    ) -> Self {
        let hidden_chans = (in_chans as f64 * expand_ratio) as usize;

        Self {
            in_chans,
            out_chans,
            // 1x1 expansion
            conv1: Conv2dBN::new(in_chans, hidden_chans, 1, 1, 0, 1, 1, device),
            act1: Gelu::new(),
            // 3x3 depthwise (groups = channels)
            conv2: Conv2dBN::new(
                hidden_chans,
                hidden_chans,
                3,
                1,
                1,
                1,
                hidden_chans, // depthwise
                device,
            ),
            act2: Gelu::new(),
            // 1x1 projection
            conv3: Conv2dBN::new(hidden_chans, out_chans, 1, 1, 0, 1, 1, device),
            act3: Gelu::new(),
            drop_path: DropoutConfig::new(drop_path).init(),
        }
    }

    /// Forward pass through MBConv
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shortcut = x.clone();

        // Expansion
        let mut out = self.conv1.forward(x);
        out = self.act1.forward(out);

        // Depthwise
        out = self.conv2.forward(out);
        out = self.act2.forward(out);

        // Projection
        out = self.conv3.forward(out);

        // Drop path
        out = self.drop_path.forward(out);

        // Residual connection (only if dimensions match)
        if self.in_chans == self.out_chans {
            out = out + shortcut;
        }

        out = self.act3.forward(out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        modeling::image_encoder::tiny_vit::test_helpers::{
            create_conv2d_bn_dict, save_module_with_bn_fix,
        },
        python::python_data::{random_python_tensor, PythonData},
        tests::helpers::{load_module, TestBackend, TEST_ALMOST_THRESHOLD},
    };
    use pyo3::{types::PyAnyMethods, PyResult, Python};

    #[test]
    fn test_mbconv_basic() {
        const FILE: &str = "tiny_vit_mbconv_basic";

        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::attach(|py| {
                use crate::python::python_data::set_seed;
                set_seed(py, 42)?;

                let torch_nn = py.import("torch.nn")?;

                // MBConv: in=64, out=64, expand_ratio=4.0, drop_path=0
                let in_chans = 64;
                let out_chans = 64;
                let hidden = 256; // 64 * 4

                // Create nested structure matching Rust
                let root = torch_nn.getattr("ModuleDict")?.call0()?;

                // conv1: 1x1 expansion
                let conv1 = create_conv2d_bn_dict(py, in_chans, hidden, 1, 1, 0, 1)?;
                root.call_method1("__setitem__", ("conv1", conv1))?;

                root.call_method1("__setitem__", ("act1", torch_nn.getattr("GELU")?.call0()?))?;

                // conv2: 3x3 depthwise
                let conv2 = create_conv2d_bn_dict(py, hidden, hidden, 3, 1, 1, hidden)?;
                root.call_method1("__setitem__", ("conv2", conv2))?;

                root.call_method1("__setitem__", ("act2", torch_nn.getattr("GELU")?.call0()?))?;

                // conv3: 1x1 projection
                let conv3 = create_conv2d_bn_dict(py, hidden, out_chans, 1, 1, 0, 1)?;
                root.call_method1("__setitem__", ("conv3", conv3))?;

                root.call_method1("__setitem__", ("act3", torch_nn.getattr("GELU")?.call0()?))?;

                // DropPath is Identity with drop_path=0
                root.call_method1(
                    "__setitem__",
                    ("drop_path", torch_nn.getattr("Identity")?.call0()?),
                )?;

                save_module_with_bn_fix(py, FILE, &root, &["conv1", "conv2", "conv3"])?;

                // Input: [1, 64, 16, 16]
                let input = random_python_tensor(py, [1, 64, 16, 16])?;

                let torch = py.import("torch")?;
                let no_grad = torch.call_method0("no_grad")?;
                let _guard = no_grad.call_method0("__enter__")?;

                // Manual forward
                let mut out = input.clone();

                // conv1 + act1
                let conv1_dict = root.call_method1("__getitem__", ("conv1",))?;
                let c = conv1_dict.call_method1("__getitem__", ("c",))?;
                let bn = conv1_dict.call_method1("__getitem__", ("bn",))?;
                out = c.call1((out,))?;
                out = bn.call1((out,))?;
                let act1 = root.call_method1("__getitem__", ("act1",))?;
                out = act1.call1((out,))?;

                // conv2 + act2
                let conv2_dict = root.call_method1("__getitem__", ("conv2",))?;
                let c = conv2_dict.call_method1("__getitem__", ("c",))?;
                let bn = conv2_dict.call_method1("__getitem__", ("bn",))?;
                out = c.call1((out,))?;
                out = bn.call1((out,))?;
                let act2 = root.call_method1("__getitem__", ("act2",))?;
                out = act2.call1((out,))?;

                // conv3
                let conv3_dict = root.call_method1("__getitem__", ("conv3",))?;
                let c = conv3_dict.call_method1("__getitem__", ("c",))?;
                let bn = conv3_dict.call_method1("__getitem__", ("bn",))?;
                out = c.call1((out,))?;
                out = bn.call1((out,))?;

                // drop_path (Identity)
                let drop_path = root.call_method1("__getitem__", ("drop_path",))?;
                out = drop_path.call1((out,))?;

                // Residual (in_chans == out_chans)
                let torch = py.import("torch")?;
                out = torch.call_method1("add", (out, &input))?;

                // act3
                let act3 = root.call_method1("__getitem__", ("act3",))?;
                out = act3.call1((out,))?;

                Ok((input.try_into()?, out.try_into()?))
            })
        }

        let (input, python_output) = python().unwrap();
        let device = Default::default();

        let mut mbconv = MBConv::<TestBackend>::new(
            64,  // in_chans
            64,  // out_chans
            4.0, // expand_ratio
            0.0, // drop_path
            &device,
        );

        mbconv = load_module(FILE, mbconv);

        let output = mbconv.forward(input.into());

        python_output.almost_equal(output, Some(TEST_ALMOST_THRESHOLD));
    }
}
