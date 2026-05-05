use ndarray::{Axis, Array1, Array2, ArrayView2};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
//to keep randomly generated weights the same each time for testing stability
use rand::SeedableRng; 
use rand::rngs::StdRng;

use crate::dataset::DataSet;


pub struct NeuralNetwork {
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
}

impl NeuralNetwork {
    pub fn new(layers: &[usize]) -> Self {

        let mut weights= Vec::new();
        let mut biases = Vec::new();

        let mut rng = StdRng::seed_from_u64(69);

        for window in layers.windows(2) {
            let input_size = window[0];
            let output_size = window[1];

            let he_scaling = (2.0 / input_size as f32).sqrt();

            let mut w = Array2::random_using(
                            (output_size, input_size),StandardNormal, &mut rng);
            w *= he_scaling;

            let b = Array1::zeros(output_size);

            weights.push(w);
            biases.push(b);
        }
    
        Self {weights,biases}
    }

    //google this function. i can't explain it here
    pub fn soft_max(mut z: Array2<f32>) -> Array2<f32> {

        /*we want to find the max value of the passed array because we will .exp() it. f32 has limit e38 which can be easily
        exceed by an exponential function. By subtracting every element by the max, we ensure that all values are below or equal
        to 0. It's okay for underflow to happen because it will just be 0.0 and math still works. We can't overflow though or else
        it becomes infinity. */
        let max_z = z.fold_axis(Axis(1),f32::NEG_INFINITY, |&a, &b| a.max(b));

        let max_z_col = max_z.insert_axis(Axis(1));
        z -= &max_z_col;

        z.mapv_inplace(|val| val.exp());
        
        let sum_z_col = z.sum_axis(Axis(1)).insert_axis(Axis(1));
        z /= &sum_z_col;

        z

    }

    /* we use arrayview because orginally i was passing a slice to this function. i changed after i shuffled the 
    data each epoch but i'm too lazy to change it. i doubt it adds any overhead though */
    pub fn forwardpass(&self, x: ArrayView2<f32>) -> Vec<Array2<f32>> {
/* not technically the number of layers. think of it as the number of layers weights and biases are applied to.
bias is not applied to input layer and weights are not applied to output layer */
        let num_layers = self.weights.len();
        let mut activations = Vec::with_capacity(num_layers + 1);

        activations.push(x.clone().to_owned());
        let mut a = x.clone().to_owned();

        for i in 0..num_layers {
            let mut z = a.dot(&self.weights[i].t()) + &self.biases[i]; 
            //bias dimensions technically don't match up but ndarrays are smart so it deals with it automatically

            if i < num_layers - 1 {
                //ReLU activation function
                z.mapv_inplace(|val| val.max(0.0));
                a = z;
            } else {
                //softmax for last layer (prediction layer)
                a = Self::soft_max(z);
            }
            activations.push(a.clone());
        }
        activations
    }


//gradients explode when learning rates > 0.05 and i'm not sure why
    pub fn calculate_gradients(&mut self, x: ArrayView2<f32>, y: ArrayView2<f32>, learning_rate: f32) {
//dimension of x is [batch_size, 784] and y is [batch_size, 10]
        let batch_size = x.nrows() as f32;
        //same num_layers as in forward pass
        let num_layers = self.weights.len();

        let activations = self.forwardpass(x);  
//derivative of softmax and cross entropy loss simplifies nicely            
        let mut dz = &activations[num_layers] - &y;


//rev() for backprop. the back means backward
        for i in (0..num_layers).rev() {

            let a_prev = &activations[i];

/* [dz] = [batch_size, out], [a_prev] = [batch_size,in], [weights[i]] = [out, in]
[dz]^T = [out, batchsize] so dw = [dz]^T[a_prev] = [out, bs][bs,in] = [out,in]
Preserving dimensions is an easy way to keep track of how to multiply but it's not a math proof for why we do it this way (i think) */
            let dw = dz.t().dot(a_prev);
            let db = dz.sum_axis(Axis(0));

/* Scale learning rate here for we don't have to individually find the mean for dw and db. saves an extra computation
    learning_rate (or lr) is almost always >= 0.0001, batch_size = 64 usually but let's say batch_size <= 1,000
    then scaled_lr is 1e-7 while f32 underflows at ~1.75e-38
    hence, we're safe from underflow and scaling our gradients by 0.0
    if we were to use f16 or f8 though there would be danger, however */
            let scaled_lr = learning_rate / batch_size;
// negative scale because we want to minimize loss. so move in opposite direction of gradient
            self.weights[i].scaled_add(-scaled_lr, &dw);
            self.biases[i].scaled_add(-scaled_lr, &db);

            if i > 0 {
/*dot prev weights with prev error (dz), multiply by ReLu derivative by chain rule
[dz] = [batchsize, out], [weights] = [out, in] [next_dz] = [dz][wights] = [batchsize, in]
[a_prev] = [batchsize, in] 
ReLU derivative is 0.0 if input is 0.0 and 1.0 otherwise.
Since we element-wise multiply everything, we can check the value of a_prev and set the corresponding value of
[next_dz] to 0.0 (multiply by 0.0) or do nothing (multiply by 1.0)
*/
                let mut next_dz = dz.dot(&self.weights[i]);
                next_dz.zip_mut_with(a_prev, |error, &act| 
                    {if act<=0.0 {*error = 0.0}});
//next_dz is simply for ease of understanding
                dz = next_dz;
            }

        }
    }

    pub fn evaluate(&self, test_x: &Array2<f32>, test_y: &Array2<f32>) -> f32 {

        let activations = self.forwardpass(test_x.view());
        let predictions = activations.last().unwrap(); //dimension looks like [10000, 10]

        let mut correct = 0;
        let num_samples = test_x.nrows();

//loop through 10,000 samples here because i got lazy. also because forwardpass takes way more computation anyway and i'm not sure this will change much
        for i in 0..num_samples {
            let pred_row = predictions.row(i);
            let label_row = test_y.row(i);

//iterate through the array -> find the index of the highest value -> return that index.
            let predicted_digit = pred_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();

            let actual_digit = label_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap();
//definitely did not have to do this one by one but im tired at this point
            if predicted_digit == actual_digit {
                correct += 1;
            }
        }

        correct as f32 / num_samples as f32
    }

//rust doesnt support auto initialized values so that's annoying. set lr to 0.01, batch_size to 64, epochs to whatever you desire
    pub fn fit(&mut self, dataset: &DataSet, mut learning_rate: f32, batch_size: usize, epochs: i32) {
        
        for epoch in 0..epochs {
//sounds good in theory to scale lr down but not sure if this did anything in practice
            if epoch % 10 == 0 {
                learning_rate *= 0.95;
            }
//instead of shuffling the entire dataset, we can just shuffle the indices
            let indices = dataset.get_shuffled_indices();

            for chunk in indices.chunks(batch_size) {

                let batch_x = dataset.train_x.select(Axis(0), chunk);
                let batch_y = dataset.train_y.select(Axis(0), chunk);

                self.calculate_gradients(batch_x.view(), batch_y.view(), learning_rate);
            }
            let acc = self.evaluate(&dataset.test_x, &dataset.test_y);

            println!("Epoch: {:>2} | Test Accuracy: {:>6.4}%", epoch + 1, acc);

        }
    }
            
}