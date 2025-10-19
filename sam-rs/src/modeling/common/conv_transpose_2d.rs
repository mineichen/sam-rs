use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, module::conv_transpose2d, ops::ConvTransposeOptions, Tensor},
};

pub struct ConvTranspose2dConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    padding_out: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    bias: bool,
}
impl ConvTranspose2dConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: [usize; 2]) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: [1, 1],
            padding: [0, 0],
            padding_out: [0, 0],
            dilation: [1, 1],
            groups: 1,
            bias: true,
        }
    }
    pub fn set_stride(&mut self, stride: [usize; 2]) -> &mut Self {
        self.stride = stride;
        self
    }
    pub fn set_padding(&mut self, padding: [usize; 2]) -> &mut Self {
        self.padding = padding;
        self
    }
    pub fn set_padding_out(&mut self, padding_out: [usize; 2]) -> &mut Self {
        self.padding_out = padding_out;
        self
    }
    pub fn set_dilation(&mut self, dilation: [usize; 2]) -> &mut Self {
        self.dilation = dilation;
        self
    }
    pub fn set_groups(&mut self, groups: usize) -> &mut Self {
        self.groups = groups;
        self
    }
    pub fn set_bias(&mut self, bias: bool) -> &mut Self {
        self.bias = bias;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTranspose2d<B> {
        ConvTranspose2d {
            weight: Param::from_tensor(Tensor::ones(
                [
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ],
                device,
            )),
            bias: match self.bias {
                true => Some(Param::from_tensor(Tensor::ones(
                    [self.out_channels],
                    device,
                ))),
                false => None,
            },
            stride: self.stride,
            padding2: self.padding,
            padding_out: self.padding_out,
            dilation: [1, 1],
            groups: 1,
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvTranspose2d<B: Backend> {
    pub weight: Param<Tensor<B, 4>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    stride: [usize; 2],
    padding2: [usize; 2],
    padding_out: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl<B: Backend> ConvTranspose2d<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let weight = self.weight.val();

        let res: Tensor<B, 4> = conv_transpose2d(
            x,
            weight,
            match &self.bias {
                Some(bias) => Some(bias.val()),
                None => None,
            },
            ConvTransposeOptions::new(
                self.stride,
                self.padding2,
                self.padding_out,
                self.dilation,
                self.groups,
            ),
        );
        res
    }
}

#[cfg(test)]
mod test {
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    use super::ConvTranspose2dConfig;
    type Backend = NdArray<f32>;

    #[test]
    fn test_conv_transpose_2d() {
        // Params
        let i: usize = 64;
        let o: usize = 16;
        let k: usize = 2;
        let stride = 2;

        let device = Default::default();
        let burn_conv = ConvTranspose2dConfig::new(i, o, [k, k])
            .set_stride([stride, stride])
            .init::<Backend>(&device);

        let shape: [usize; 4] = [16, i, 16, 16];

        let burn_input =
            Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let burn_output = burn_conv.forward(burn_input);

        // Check that output has correct shape
        let expected_output_shape = [16, o, 32, 32]; // stride=2 doubles the size
        assert_eq!(burn_output.dims(), expected_output_shape);
    }
}
