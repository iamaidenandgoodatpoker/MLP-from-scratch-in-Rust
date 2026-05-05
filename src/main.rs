mod network;
mod dataset;
use network::NeuralNetwork;
use dataset::DataSet;
extern crate accelerate_src; //comment this out if you're not on mac

fn main() {
    //you can experiment with different structures but i found this one works the best.
    let mut model = NeuralNetwork::new(&[784, 128, 10]);

    //gradients explode when learning_rate is over 0.05 (sometimes less) and i havent fixed that issue yet
    let dataset = DataSet::load_mnist();
    let batch_size = 64;
    let learning_rate = 0.01;
    let epochs = 100;

    //how nice! i put everything into one little function. main() is so clean.
    model.fit(&dataset, learning_rate, batch_size, epochs);

}
