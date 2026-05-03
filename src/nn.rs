use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

pub struct NeuralNetwork {
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
}

impl NeuralNetwork {
    pub fn new(layers: &[usize]) -> Self {

        let mut weights= Vec::new();
        let mut biases = Vec::new();

        for window in layers.windows(2) {
            let input_size = window[0];
            let output_size = window[1];
            let he_scaling = (2.0 / input_size as f32).sqrt();

            let mut w = Array2::random((output_size, input_size),StandardNormal);
            w *= he_scaling;

            let b = Array1::zeros(output_size);

            weights.push(w);
            biases.push(b);
        }
    
        Self {weights,biases}
    }

    pub fn forwardpass() {
        
    }
}