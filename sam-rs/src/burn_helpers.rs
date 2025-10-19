use burn::tensor::{
    backend::Backend, BasicOps, ElementConversion, Int, Tensor, TensorData, TensorKind,
};

pub trait TensorHelpers<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>> {
    fn calc_dims<const D2: usize>(&self, dims: [usize; D2]) -> [usize; D2];

    fn unsqueeze_end<const D2: usize>(self) -> Tensor<B, D2, K>;
    fn reshape_max<const D2: usize>(&self, dims: [usize; D2]) -> Tensor<B, D2, K>;

    fn of_slice<T: burn::tensor::Element>(
        slice: Vec<T>,
        shape: [usize; D],
        device: &B::Device,
    ) -> Self
    where
        K::Elem: ElementConversion;
    fn to_slice<T: burn::tensor::Element>(&self) -> (Vec<T>, [usize; D])
    where
        K::Elem: ElementConversion;
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
    fn of_slice<T: burn::tensor::Element>(
        slice: Vec<T>,
        shape: [usize; D],
        device: &B::Device,
    ) -> Self
    where
        K::Elem: ElementConversion,
    {
        let slice: Vec<K::Elem> = slice.into_iter().map(|x| K::Elem::from_elem(x)).collect();
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
}

pub trait ToFloat<B: Backend, const D: usize> {
    fn to_float(&self) -> Tensor<B, D>;
}
impl<B: Backend, const D: usize> ToFloat<B, D> for Tensor<B, D, Int> {
    fn to_float(&self) -> Tensor<B, D> {
        let device = self.device();
        let (slice, shape) = self.to_slice::<f32>();
        Tensor::of_slice(slice, shape, &device)
    }
}
