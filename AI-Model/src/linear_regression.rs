//Simple Linear Regression implementation in Rust
//module now tested, could have some more testing for edge cases
//Works relatively well so far.

///Q: Why Linear Regression?
///A: As mentioned in the Project Plan and elsewhere, linear regression, besides being low-risk, also has the advantage of working well with continuous data such as mean Dark Channel.

//TODO: further testing if necessary, extension to more variables, etc.

use ndarray::{Array1, Array2};

pub struct LinearRegression {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl LinearRegression {
    pub fn new(num_features: usize) -> Self {
        Self {
            weights: Array1::zeros(num_features),
            bias: 0.0,
        }
    }

    //Predict output given input matrix and current weights/biases of model, simple y = mx + b implementation
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        features.dot(&self.weights) + self.bias
    }

    /*
    Gradient descent algorithm for training, implemented manually
    @param features: matrix of [samples, features]
    @param targets: vector of [samples]
    @param learning_rate: multiplier for how much to adjust weights/biases each iteration. Higher means faster convergence on the right weights/bias, but more prone to overshooting the optimal solution
    @param epochs: number of iterations through the dataset
     */
    pub fn train(&mut self, features: &Array2<f64>, targets: &Array1<f64>, learning_rate: f64, epochs: usize, ) {
        let n_samples = features.nrows() as f64;

        for epoch in 0..epochs {
            let predictions = self.predict(features); //initial "guess" of output based on current weights/biases
            let errors = targets - &predictions;

            let w_grad = -(2.0 / n_samples) * features.t().dot(&errors); //weight gradient = (-2/N) * x_transposed * error
            let b_grad = -(2.0 / n_samples) * errors.sum(); //bias gradient = (-2/N) * sum(error)

            self.weights = &self.weights - &(w_grad * learning_rate); //update based on sensitivity multiplier of learning_rate
            self.bias = self.bias - (b_grad * learning_rate);

            if epoch % 20 == 0 {
                let mse = errors.mapv(|e| e.powi(2)).mean().unwrap();
                println!("Epoch {}: MSE Loss = {:.4}", epoch, mse); //prints Mean Squared Error loss every 20 epochs to update user on convergence progress
            }
        }
    }
}