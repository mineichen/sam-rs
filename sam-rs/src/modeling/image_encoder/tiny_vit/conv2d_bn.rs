use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

/// Conv2d + BatchNorm2d block
/// Equivalent to PyTorch's Conv2d_BN from TinyViT
///
/// This is the fundamental building block of TinyViT, combining a 2D convolution
/// with batch normalization. The convolution has no bias since BatchNorm handles it.
#[derive(Debug, Module)]
pub struct Conv2dBN<B: Backend> {
    pub c: Conv2d<B>,
    pub bn: BatchNorm<B, 2>,
}

impl<B: Backend> Conv2dBN<B> {
    /// Create a new Conv2d_BN layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Kernel size (square kernel)
    /// * `stride` - Stride (default 1)
    /// * `padding` - Padding (default 0)
    /// * `dilation` - Dilation (default 1)
    /// * `groups` - Number of groups for grouped convolution (default 1)
    /// * `device` - Device to create the layer on
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        device: &B::Device,
    ) -> Self {
        let c = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .with_dilation([dilation, dilation])
            .with_groups(groups)
            .with_bias(false) // No bias when using BatchNorm
            .init(device);

        let bn = BatchNormConfig::new(out_channels).init(device);

        Self { c, bn }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.c.forward(x);
        self.bn.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        modeling::image_encoder::tiny_vit::test_helpers::save_module_with_bn_fix,
        python::python_data::{random_python_tensor, PythonData},
        tests::helpers::{load_module, TestBackend},
    };
    use pyo3::{types::PyAnyMethods, PyResult, Python};

    #[test]
    fn test_conv2d_bn_basic() {
        const FILE: &str = "tiny_vit_conv2d_bn_basic";

        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::attach(|py| {
                crate::python::python_data::init_torch(py, 42)?;

                // Create Conv2d_BN structure that matches Rust's field names
                let torch_nn = py.import("torch.nn")?;

                use pyo3::types::PyDict;
                let kwargs = PyDict::new(py);
                kwargs.set_item("bias", false)?; // No bias when using BatchNorm

                // Create Conv2d without bias: in=3, out=16, ks=3, stride=2, padding=1
                let conv = torch_nn
                    .getattr("Conv2d")?
                    .call((3, 16, 3, 2, 1), Some(&kwargs))?;

                // Create BatchNorm2d
                let bn = torch_nn.getattr("BatchNorm2d")?.call1((16,))?;

                // Create a ModuleDict with field names matching Rust struct
                let module_dict = torch_nn.getattr("ModuleDict")?.call0()?;
                module_dict.call_method1("__setitem__", ("c", conv))?;
                module_dict.call_method1("__setitem__", ("bn", bn))?;

                save_module_with_bn_fix(py, FILE, &module_dict, &[""])?;

                // Create input: [1, 3, 8, 8]
                let input = random_python_tensor(py, [1, 3, 8, 8])?;

                // Run forward pass manually (ModuleDict doesn't have forward)
                let torch = py.import("torch")?;
                let no_grad = torch.call_method0("no_grad")?;
                let _guard = no_grad.call_method0("__enter__")?;

                let c = module_dict.call_method1("__getitem__", ("c",))?;
                let bn = module_dict.call_method1("__getitem__", ("bn",))?;
                let x = c.call1((&input,))?;
                let output = bn.call1((x,))?;

                Ok((input.try_into()?, output.try_into()?))
            })
        }

        let (input, python_output) = python().unwrap();
        let device = Default::default();

        // Create Rust version with same parameters
        let mut conv_bn = Conv2dBN::<TestBackend>::new(
            3,  // in_channels
            16, // out_channels
            3,  // kernel_size
            2,  // stride
            1,  // padding
            1,  // dilation
            1,  // groups
            &device,
        );

        // Load weights from Python
        conv_bn = load_module(FILE, conv_bn);

        // Forward pass
        let output = conv_bn.forward(input.into());

        // Compare with Python output
        python_output.almost_equal(output, None);
    }

    #[test]
    fn test_conv2d_bn_depthwise() {
        const FILE: &str = "tiny_vit_conv2d_bn_depthwise";

        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::attach(|py| {
                crate::python::python_data::init_torch(py, 42)?;

                // Create depthwise convolution (groups=channels)
                let torch_nn = py.import("torch.nn")?;

                use pyo3::types::PyDict;
                let kwargs = PyDict::new(py);
                kwargs.set_item("groups", 32)?; // Depthwise: groups=in_channels
                kwargs.set_item("bias", false)?;

                // Create grouped Conv2d: in=32, out=32, ks=3, stride=1, padding=1, groups=32
                let conv = torch_nn
                    .getattr("Conv2d")?
                    .call((32, 32, 3, 1, 1), Some(&kwargs))?;

                // Create BatchNorm2d
                let bn = torch_nn.getattr("BatchNorm2d")?.call1((32,))?;

                // Create ModuleDict
                let module_dict = torch_nn.getattr("ModuleDict")?.call0()?;
                module_dict.call_method1("__setitem__", ("c", conv))?;
                module_dict.call_method1("__setitem__", ("bn", bn))?;

                save_module_with_bn_fix(py, FILE, &module_dict, &[""])?;

                let input = random_python_tensor(py, [1, 32, 8, 8])?;

                let torch = py.import("torch")?;
                let no_grad = torch.call_method0("no_grad")?;
                let _guard = no_grad.call_method0("__enter__")?;

                let c = module_dict.call_method1("__getitem__", ("c",))?;
                let bn = module_dict.call_method1("__getitem__", ("bn",))?;
                let x = c.call1((&input,))?;
                let output = bn.call1((x,))?;

                Ok((input.try_into()?, output.try_into()?))
            })
        }

        let (input, python_output) = python().unwrap();
        let device = Default::default();

        // Create depthwise convolution in Rust
        let mut conv_bn = Conv2dBN::<TestBackend>::new(
            32, // in_channels
            32, // out_channels
            3,  // kernel_size
            1,  // stride
            1,  // padding
            1,  // dilation
            32, // groups (depthwise)
            &device,
        );

        conv_bn = load_module(FILE, conv_bn);

        let output = conv_bn.forward(input.into());

        python_output.almost_equal(output, None);
    }

    #[test]
    fn test_conv2d_bn_1x1() {
        const FILE: &str = "tiny_vit_conv2d_bn_1x1";

        fn python() -> PyResult<(PythonData<4>, PythonData<4>)> {
            Python::attach(|py| {
                crate::python::python_data::init_torch(py, 42)?;

                // Test 1x1 convolution (point-wise, common in TinyViT)
                let torch_nn = py.import("torch.nn")?;

                use pyo3::types::PyDict;
                let kwargs = PyDict::new(py);
                kwargs.set_item("bias", false)?;

                // Create 1x1 Conv2d: in=64, out=128, ks=1, stride=1, padding=0
                let conv = torch_nn
                    .getattr("Conv2d")?
                    .call((64, 128, 1, 1, 0), Some(&kwargs))?;

                let bn = torch_nn.getattr("BatchNorm2d")?.call1((128,))?;

                let module_dict = torch_nn.getattr("ModuleDict")?.call0()?;
                module_dict.call_method1("__setitem__", ("c", conv))?;
                module_dict.call_method1("__setitem__", ("bn", bn))?;

                save_module_with_bn_fix(py, FILE, &module_dict, &[""])?;

                let input = random_python_tensor(py, [2, 64, 16, 16])?;

                let torch = py.import("torch")?;
                let no_grad = torch.call_method0("no_grad")?;
                let _guard = no_grad.call_method0("__enter__")?;

                let c = module_dict.call_method1("__getitem__", ("c",))?;
                let bn = module_dict.call_method1("__getitem__", ("bn",))?;
                let x = c.call1((&input,))?;
                let output = bn.call1((x,))?;

                Ok((input.try_into()?, output.try_into()?))
            })
        }

        let (input, python_output) = python().unwrap();
        let device = Default::default();

        let mut conv_bn = Conv2dBN::<TestBackend>::new(
            64,  // in_channels
            128, // out_channels
            1,   // kernel_size (1x1)
            1,   // stride
            0,   // padding
            1,   // dilation
            1,   // groups
            &device,
        );

        conv_bn = load_module(FILE, conv_bn);

        let output = conv_bn.forward(input.into());

        python_output.almost_equal(output, None);
    }
}
