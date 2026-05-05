mod network;
mod dataset;
use network::NeuralNetwork;
use dataset::DataSet;

fn main() {
    let mut model = NeuralNetwork::new(&[784, 128, 10]);

    let dataset = DataSet::load_mnist();
    let batch_size = 64;
    let learning_rate = 0.05;
    let epochs = 100;

    model.fit(&dataset, learning_rate, batch_size, epochs);

}
