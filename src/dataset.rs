use mnist::{Mnist, MnistBuilder};
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DataSet {
    pub train_x: Array2<f32>,
    pub train_y: Array2<f32>,
    pub test_x: Array2<f32>,
    pub test_y: Array2<f32>,
}

impl DataSet {
    pub fn load_mnist() -> Self {
        let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
            .base_path("data") //folder where i have all the mnist datasets
            .label_format_one_hot()
            .finalize();

        let train_x = Array2::from_shape_vec((60_000, 784), trn_img).unwrap().mapv(|x| x as f32 / 255.0);
        let train_y = Array2::from_shape_vec((60_000, 10), trn_lbl).unwrap().mapv(|x| x as f32);

        let test_x = Array2::from_shape_vec((10_000, 784), tst_img).unwrap().mapv(|x| x as f32 / 255.0);
        let test_y = Array2::from_shape_vec((10_000, 10), tst_lbl).unwrap().mapv(|x| x as f32);

        DataSet{ train_x, train_y, test_x, test_y }
    }
    pub fn get_shuffled_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.train_x.nrows()).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
        indices
    }
}