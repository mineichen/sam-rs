use burn::tensor::{backend::Backend, BasicOps, ElementConversion, Tensor, TensorData, TensorKind};

pub trait TensorHelpers<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>> {
    fn calc_dims<const D2: usize>(&self, dims: [usize; D2]) -> [usize; D2];

    fn unsqueeze_end<const D2: usize>(self) -> Tensor<B, D2, K>;
    fn reshape_max<const D2: usize>(&self, dims: [usize; D2]) -> Tensor<B, D2, K>;

    fn collect_shaped<T: burn::tensor::Element>(
        slice: impl IntoIterator<Item = T>,
        shape: [usize; D],
        device: &B::Device,
    ) -> Self
    where
        K::Elem: ElementConversion;
    fn to_slice<T: burn::tensor::Element>(&self) -> (Vec<T>, [usize; D])
    where
        K::Elem: ElementConversion;

    fn repeat_interleave(&self, repeats: usize, dim: usize) -> Tensor<B, D, K>;
}
impl<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>> TensorHelpers<B, D, K>
    for Tensor<B, D, K>
{
    fn calc_dims<const D2: usize>(&self, dims: [usize; D2]) -> [usize; D2] {
        let max_count = dims.iter().filter(|&&x| x == usize::MAX).count();
        assert!(
            max_count <= 1,
            "There mustca be exactly one usize::MAX in the dims array"
        );
        if max_count == 0 {
            return dims;
        }
        let elems = self.dims().iter().fold(1, |acc, &x| acc * x)
            / dims
                .iter()
                .filter(|x| **x != usize::MAX)
                .fold(1, |acc, &x| acc * x);
        dims.map(|x| if x == usize::MAX { elems } else { x })
    }
    fn reshape_max<const D2: usize>(&self, dims: [usize; D2]) -> Tensor<B, D2, K> {
        self.clone().reshape(self.calc_dims(dims))
    }

    fn unsqueeze_end<const D2: usize>(self) -> Tensor<B, D2, K> {
        let tensor = self.unsqueeze();
        let mut dims = [0; D2];
        let diff = D2 - D;
        for i in 0..D2 {
            dims[i] = match i < D {
                true => i + diff,
                false => i - D,
            } as isize
        }
        let tensor = tensor.permute(dims);
        tensor
    }
    fn collect_shaped<T: burn::tensor::Element>(
        slice: impl IntoIterator<Item = T>,
        shape: [usize; D],
        device: &B::Device,
    ) -> Self
    where
        K::Elem: ElementConversion,
    {
        let slice = slice.into_iter().map(|x| K::Elem::from_elem(x)).collect();
        let data = TensorData::new(slice, shape);
        Tensor::from_data(data, device)
    }
    fn to_slice<T: burn::tensor::Element>(&self) -> (Vec<T>, [usize; D])
    where
        K::Elem: ElementConversion,
    {
        let data = self.to_data();
        let slice: Vec<T> = data
            .as_slice::<K::Elem>()
            .unwrap()
            .iter()
            .map(|x| x.elem())
            .collect();
        let shape: [usize; D] = data.shape.try_into().expect("Shape dimension mismatch");
        (slice, shape)
    }

    fn repeat_interleave(&self, repeats: usize, dim: usize) -> Tensor<B, D, K> {
        let shape = self.dims();
        let mut new_shape = shape;
        new_shape[dim] *= repeats;

        let data = self.to_data();
        let slice = data.as_slice::<K::Elem>().unwrap();

        let mut result = Vec::with_capacity(new_shape.iter().product());

        let outer_size: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();

        for outer in 0..outer_size {
            for d in 0..dim_size {
                for _ in 0..repeats {
                    let start = outer * dim_size * inner_size + d * inner_size;
                    result.extend_from_slice(&slice[start..start + inner_size]);
                }
            }
        }

        Tensor::from_data(TensorData::new(result, new_shape), &self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    type TestBackend = NdArray<f32>;

    #[test]
    fn test_repeat_interleave_1d() {
        // PyTorch: torch.tensor([1, 2, 3]).repeat_interleave(2) -> [1, 1, 2, 2, 3, 3]
        let device = Default::default();
        let t: Tensor<TestBackend, 1> = Tensor::collect_shaped([1.0f32, 2.0, 3.0], [3], &device);
        let result = t.repeat_interleave(2, 0);
        let (vals, _) = result.to_slice::<f32>();
        assert_eq!(vals, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_interleave_2d_dim0() {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).repeat_interleave(2, dim=0)
        // -> [[1, 2], [1, 2], [3, 4], [3, 4]]
        let device = Default::default();
        let t: Tensor<TestBackend, 2> =
            Tensor::collect_shaped([1.0f32, 2.0, 3.0, 4.0], [2, 2], &device);
        let result = t.repeat_interleave(2, 0);
        let (vals, shape) = result.to_slice::<f32>();
        assert_eq!(shape, [4, 2]);
        assert_eq!(vals, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_interleave_2d_dim1() {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).repeat_interleave(2, dim=1)
        // -> [[1, 1, 2, 2], [3, 3, 4, 4]]
        let device = Default::default();
        let t: Tensor<TestBackend, 2> =
            Tensor::collect_shaped([1.0f32, 2.0, 3.0, 4.0], [2, 2], &device);
        let result = t.repeat_interleave(2, 1);
        let (vals, shape) = result.to_slice::<f32>();
        assert_eq!(shape, [2, 4]);
        assert_eq!(vals, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn test_repeat_interleave_4d() {
        // Test 4D tensor (like image embeddings [1, 256, 64, 64])
        let device = Default::default();
        let t: Tensor<TestBackend, 4> = Tensor::collect_shaped(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1, 2, 2, 2],
            &device,
        );
        // repeat_interleave on batch dim
        let result = t.repeat_interleave(2, 0);
        assert_eq!(result.dims(), [2, 2, 2, 2]);
        let (vals, _) = result.to_slice::<f32>();
        // First batch repeated, then second
        assert_eq!(vals[..8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(vals[8..], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}
