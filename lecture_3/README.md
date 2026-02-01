# DL regression

Write a DL regression model using keras with the following steps:

**Data Loading**

1. Download the data with:
    ```
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
    ```
    This file contains a list of observed miles per gallon performance (MPG) for multiple car models following the convention `['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']`.

**Data preprocessing**

2. Clean the data for unknown values, such as empty entries.

3. Split the data into training and test sets using pandas sample fraction of 0.8.

4. Inspect the data using correlation plot.

5. Allocate a normalization layer with `tensorflow.keras.layers.Normalization`.

**Linear regressor**

6. Perform a linear regression fit using just `Horsepower` as input (1 input). For this exercise you can use `mean_absolute_error` as loss function, Adam for the optimization and 100 epochs. Use `validation_split=0.2` when fitting. Plot the loss function history.

7. Repeat the previous point but now using all 9 input variables. Plot the loss function history.

**DNN model fit**

8. Build a deep neural network model with 2 layers containing 64 nodes each and `relu` activation function and a last layer with a single unit and linear activation function.Perform a fit following the setup from 6 (single input).

9. Repeat 8 with now all 9 input variables.

10. Compare the mean absolute error for the test set. Make prediction using the DNN model.

# Hyperparameter scan

**Data loading**

1. Load the mnist dataset from `tensorflow.keras.datasets.mnist`. Study the dataset size (shape) and normalize the pixels.

**DNN model**

2. Design a NN architecture for the classification of all digits.

**Hyperparameter scan**

3. Define a function which parametrizes the learning rate and the number of units of the DNN model using a python dict.

4. Implement a Tree Parzen Estimator with the [hyperopt](http://hyperopt.github.io/hyperopt/) library.

5. Plot the accuracy vs learning rate and number of layers for each trial.
