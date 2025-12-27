#ifndef NEURAL_H
#define NEURAL_H

#include <Eigen/Eigen>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <sstream>
#include "main.h"

using namespace std;

typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

// neural network implementation class!
class NeuralNetwork {
private:
    vector<uint> topology; // topology
public:
    // constructor
    NeuralNetwork(vector<uint> topology, Scalar learningRate = Scalar(0.005));

    // function for forward propagation of data
    RowVector propagateForward(const RowVector& input);

    // function for backward propagation of errors made by neurons
    void propagateBackward(RowVector& output);

    // function to calculate errors made by neurons in each layer
    void calcErrors(RowVector& output);

    // function to update the weights of connections
    void updateWeights(void);

    // save the weight matrices after training
    void saveWeights(string filename);

    // load the weight matrices in order to skip training
    void loadWeights(string filenameid);

    // as a debugging tool, print the weight matrices
    void printWeights(void);

    // function to train the neural network give an array of data points
    vector<Scalar> train(vector<RowVector*> input_data, vector<RowVector*> output_data);

    // storage objects for working of neural network
    /*
          use pointers when using std::vector<Class> as std::vector<Class> calls destructor of 
          Class as soon as it is pushed back! when we use pointers it can't do that, besides
          it also makes our neural network class less heavy!! It would be nice if you can use
          smart pointers instead of usual ones like this
        */
    vector<RowVector*> neuronLayers; // stores the different layers of out network
    // vector<unique_ptr<RowVector>> neuronLayers;
    vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
    vector<RowVector*> deltas; // stores the error contribution of each neurons
    vector<Matrix*> weights; // the connection weights itself
    Scalar learningRate;

    // destructor -> free neuronLayers, cacheLayers, weights, and deltas
    ~NeuralNetwork();
};

// User function declaration
Scalar activationFunction(Scalar x);
Scalar activationFunctionDerivative(Scalar x);
#endif // NEURAL_Hfilename