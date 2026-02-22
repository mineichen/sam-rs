# STATE
The implementation creates a mask, but currently has worse quality compared to the original python version (See output of test_prediction)

## RUN tests
Tests share the Python GIL in a unsafe way. Therefore, thests must be executed with:
cargo test --release -- --test-threads=1



## Converting models

To convert models, drop the model file into `sam-convert` folder (e.g. sam-convert/sam_vit_b_01ec64.pth), and run `cargo run --release -- <vit_h | vit_b | vit_l | test> <file_name> <?skip_python>`. The converted model will be in the same folder as the original model, with the same name, but with the extension `.bin.gz`. It will take quite a long time, and needs some disk space (the middle json file is up to 16 GB for larger models).

This will:
1. Load the weights in python
2. Convert them into json a json file that can be read by the rust code 
3. Rust loads the model from that json file
4. Rust saves the model as a binary file, to minimize file size and optimize for faster load.
5. (You can delete the json file in `~/Documents/sam-models/<name>`)
