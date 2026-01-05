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

typedef enum {
    MISMATCH_IN_SIZE = -1,
    NO_ERROR = 0,
} Error_Codes_T;

class NeuralNetwork {
private:
    // function for backward propagation of errors made by neurons
    void propagateBackward(RowVector& output);

    // function to calculate errors made by neurons in each layer
    void calcErrors(RowVector& output);

    // function to update the weights of connections
    void updateWeights(void);
    
    // storage objects for working of neural network
    vector<RowVector*> neuronLayers; // stores the different layers of out network
    // an example of a usage of unique_ptrs - vector<unique_ptr<RowVector>> neuronLayers;
    vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
    vector<RowVector*> deltas; // stores the error contribution of each neurons
    vector<Matrix*> weights; // the connection weights itself
    Scalar learningRate;
    vector<uint> topology; // topology of the nn system
    RowVector inputScaling; // pre-processing the inputs
protected:
public:
    // constructor
    NeuralNetwork(const vector<uint>& topology, const RowVector& inputScaling, Scalar learningRate = Scalar(0.005));

    // function for forward propagation of data
    RowVector propagateForward(const RowVector& input);

    // save the weight matrices after training
    void saveWeights(string filename);

    // load the weight matrices in order to skip training
    int loadWeights(string filenameid);

    // as a debugging tool, print the weight matrices
    void printWeights(void);

    // function to train the neural network give an array of data points
    vector<Scalar> train(vector<RowVector*> input_data, vector<RowVector*> output_data);

    // destructor -> free neuronLayers, cacheLayers, weights, and deltas
    ~NeuralNetwork();

    // static methods
    static Scalar activationFunction(Scalar x);
    static Scalar activationFunctionDerivative(Scalar x);
    static bool float_cmp(const Scalar val_in1, const Scalar val_in2, const Scalar threshold_in);
};
#endif // NEURAL_H