# Introduction and motivation
- Make the code work from the example
- Correct the mistakes from the example
- Implementation of using the neural network standalone (no training needed, just loading the weights)
- Implementation of some kind of testing based on good practices: two data sets, one for training, one for validation
- Study the Eigen library
- Study how the data is organized
- Learn more about C++

# Overview of the algorithm
![hosted picture](https://private-user-images.githubusercontent.com/98287245/531908935-81986796-184a-4c3c-967f-b076c7327b89.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc2MDk1MDMsIm5iZiI6MTc2NzYwOTIwMywicGF0aCI6Ii85ODI4NzI0NS81MzE5MDg5MzUtODE5ODY3OTYtMTg0YS00YzNjLTk2N2YtYjA3NmM3MzI3Yjg5LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAxMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMTA1VDEwMzMyM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTg0Y2U0ODQ2M2FmM2QxYmI4NzE5YmMxNzRmZTY2NTE1ZWFmY2JhMmE1OTU5NzYwMWI4MWFhZDMxMjEwZGVjNGEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.QX0SFGIwXZS4M4AqN4cI1PrLzeYHqbgk450IXlBpebk)

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