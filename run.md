clone the repo


If you don't have mac or it doesn't work. remove the following block from cargo.toml

ndarray = { version = "0.15", features = ["blas"] }
accelerate-src = "0.3.2"

and replace it with

ndarray = "0.15"


Other than that, there are instructions in dataset.rs and main.rs for what to do.



Then in terminal run

cargo run --release
