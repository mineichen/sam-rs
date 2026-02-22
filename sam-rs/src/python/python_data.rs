use std::f32;

use burn::tensor::{backend::Backend, BasicOps, Element, ElementConversion, Tensor, TensorKind};
use pyo3::types::{PyAnyMethods, PyModule};
use pyo3::Bound;
use pyo3::{types::PyTuple, FromPyObject, PyAny, PyErr, PyResult, Python};

use crate::{burn_helpers::TensorHelpers, sam_predictor::Size};

pub trait PythonDataKind: std::fmt::Debug + PartialEq + Clone + Element + Sized + Copy {}
impl PythonDataKind for f32 {}
impl PythonDataKind for i64 {}

/// Initialize a clean Python test environment
/// This helps isolate tests by resetting random state and ensuring clean imports
pub fn init_torch(py: Python<'_>, seed: i64) -> PyResult<Bound<'_, PyModule>> {
    // Set random seed for reproducibility
    let torch = py.import("torch")?;
    torch.call_method1("manual_seed", (seed,))?;

    // Ensure numpy random seed is also set (if numpy is used)
    if let Ok(np) = py.import("numpy") {
        let np_random = np.getattr("random")?;
        np_random.call_method1("seed", (seed,))?;
    }

    Ok(torch)
}

pub fn random_python_tensor<'py, const D: usize>(
    py: Python<'py>,
    shape: [usize; D],
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let shape_tuple = pyo3::types::PyTuple::new(py, shape)?;
    let input = torch.call_method1("randn", (shape_tuple,))?;
    Ok(input.into_any())
}
pub fn random_python_tensor_int<'py, const D: usize>(
    py: Python<'py>,
    shape: [usize; D],
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let tensor = random_python_tensor(py, shape)?;
    let int = py.import("torch")?.getattr("int")?;
    let tensor = tensor.call_method1("type", (int,))?;
    Ok(tensor)
}
#[derive(PartialEq, Clone)]
pub struct PythonData<const D: usize, T: PythonDataKind = f32> {
    pub slice: Vec<T>,
    pub shape: [usize; D],
}

impl<T: PythonDataKind, const D: usize> std::fmt::Debug for PythonData<D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.slice.len();
        if len <= 10 {
            return f
                .debug_struct("TestTensor")
                .field("shape", &self.shape)
                .field("values", &self.slice)
                .finish();
        }
        f.debug_struct("TestTensor")
            .field("shape", &self.shape)
            .field("start", &self.slice[0..5].to_vec())
            .field("end", &self.slice[len - 6..len - 1].to_vec())
            .finish()
    }
}
impl<const D: usize, T: PythonDataKind> PythonData<D, T> {
    pub fn new(slice: Vec<T>, shape: [usize; D]) -> Self {
        Self { slice, shape }
    }
    pub fn equal<I: Into<Self>>(&self, other: I) {
        let other = other.into();
        assert_eq!(self, &other, "PythonData::eq failed");
    }

    #[track_caller]
    pub fn almost_equal<I: Into<Self>, X: Into<Option<f32>>>(&self, output: I, threshold: X) {
        let other: Self = output.into();
        let threshold = threshold.into().unwrap_or(1e-3);
        if self.shape != other.shape {
            panic!("TestTensor sizes don't match");
        }
        let mut exact = 0;
        let mut almost = 0;
        let mut failed = 0;
        let mut max_diff: f32 = 0.0;
        for (a, b) in self.slice.iter().zip(other.slice.iter()) {
            let a = a.to_f32();
            let b = b.to_f32();
            if a == b {
                exact += 1;
                continue;
            }
            let diff = (a - b).abs();
            if diff <= threshold {
                almost += 1;
                continue;
            };
            max_diff = max_diff.max(diff);
            failed += 1;
        }
        let total = self.slice.len();

        match failed {
            0 => {}
            _ => {
                println!("left: {:?}", self);
                println!("right: {:?}", other);
                panic!(
                    "TestTensor::eq: exact: {}, almost: {}, failed: {}, total: {}! Max threshold: {}, current: {}",
                    exact, almost, failed, total,max_diff, threshold
                );
            }
        }
    }
}

fn extract_python_data<'py, T>(data: &pyo3::Bound<'py, PyAny>) -> PyResult<Vec<T>>
where
    Vec<T>: for<'a, 'b> FromPyObject<'a, 'b, Error = PyErr>,
{
    data.getattr("flatten")?
        .call0()?
        .getattr("tolist")?
        .call0()?
        .extract()
}

impl<'py, const D: usize, T: PythonDataKind> TryFrom<pyo3::Bound<'py, PyAny>> for PythonData<D, T>
where
    Vec<T>: for<'a, 'b> FromPyObject<'a, 'b, Error = PyErr>,
{
    type Error = PyErr;
    fn try_from(data: pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        let slice = extract_python_data(&data)?;
        let shape = data.getattr("shape")?.extract::<Vec<usize>>()?;
        assert_eq!(D, shape.len(), "Shape length doesn't match");
        let shape = shape.try_into().unwrap();
        Ok(PythonData::new(slice, shape))
    }
}

pub fn pyany_to_tensor<'a, B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>>(
    data: Bound<'a, PyAny>,
) -> Tensor<B, D, K>
where
    <K as BasicOps<B>>::Elem: ElementConversion,
{
    let data: PythonData<D> = data.try_into().unwrap();
    let tensor: Tensor<B, D, K> = data.try_into().unwrap();
    tensor
}

impl<B: Backend, const D: usize, T: PythonDataKind, K: TensorKind<B> + BasicOps<B>>
    From<PythonData<D, T>> for Tensor<B, D, K>
where
    <K as BasicOps<B>>::Elem: ElementConversion,
{
    fn from(data: PythonData<D, T>) -> Self {
        let slice = data.slice;
        let shape = data.shape;
        let device = B::Device::default();
        Tensor::collect_shaped(slice, shape, &device)
    }
}

impl<B: Backend, const D: usize, T: PythonDataKind, K: TensorKind<B> + BasicOps<B>>
    From<Tensor<B, D, K>> for PythonData<D, T>
where
    <K as BasicOps<B>>::Elem: ElementConversion,
{
    fn from(data: Tensor<B, D, K>) -> Self {
        let (slice, shape) = data.to_slice();
        PythonData::new(slice, shape)
    }
}

impl<'py> TryFrom<pyo3::Bound<'py, PyAny>> for Size {
    type Error = PyErr;
    fn try_from(data: pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        let tuple = data.cast::<PyTuple>()?;
        Ok(Size(
            tuple.get_item(0)?.extract()?,
            tuple.get_item(1)?.extract()?,
        ))
    }
}
