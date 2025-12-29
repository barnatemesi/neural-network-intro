# Introduction and motivation
- Make the code work from the example
- Correct the mistakes from the example
- Implementation of using the neural network standalone (no training needed, just loading the weights)
- Implementation of some kind of testing based on good practices: two data sets, one for training, one for validation
- Study the Eigen library
- Study how the data is organized
- Learn more about C++

# Observations
There is a possibility that the algorithm does not converge. In this case, the computed weights are out of a reasonable range of expected values. When the input data is not very random, the training has a higher chance of succeeding. This is true, when training on the Kalman-filter dataset.

Works pretty well with linear equations. Even when increasing the complexity of the system, it struggles with non-linear systems (such as x^2 etc.).

# TODO
- Implementation of input processing / normalization
- Study how Gradient Descent optimization works / may be implemented
- Generate validation data for kalman-filter system
- Add new methods to save / load the weight matrices -> skip the training step when valid weight matrices are successfully loaded
- Find better training data for the kalman-filter system
- Modernize the codebase
- Use unique_ptr such as: vector<unique_ptr<RowVector>> neuronLayers. This avoids defining the class destructor.
- Figure out how to use neural networks as simple filters
- Figure out how to use neural networks as simple controllers

# Dependencies
 - Eigen library (5.0.0)
 - compiler of your choice: g++ (13.3.0) / clang (18.1.13 - works but not properly tested)
 - make (4.3)

# Sources
[geeks for geeks article](https://www.geeksforgeeks.org/machine-learning/ml-neural-network-implementation-in-c-from-scratch/)

[eigen c++ library](https://libeigen.gitlab.io/)