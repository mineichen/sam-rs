use burn::{
    module::Module,
    record::{BinGzFileRecorder, FullPrecisionSettings, Recorder},
};
use burn_ndarray::NdArray;
use sam_rs::{build_sam::SamVersion, python::recorder::load_module_from_python};
use std::{env, path::Path, time::Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        panic!("Usage: sam-convert <type> <file>");
    }
    let version = args[1].as_str();
    let file = Path::new(&args[2]);

    let version = SamVersion::from_str(version);

    let start = Instant::now();

    let sam = version.build::<NdArray<f32>>(None, &Default::default());
    let sam = load_module_from_python(sam, version, file).unwrap();

    println!("Saving module in rust...");
    let recorder = BinGzFileRecorder::<FullPrecisionSettings>::default();
    recorder.record(sam.into_record(), file.into()).unwrap();

    println!("Rust time: {:?}", start.elapsed());
}
