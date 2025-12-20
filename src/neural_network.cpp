#include "neural_network.h"

using namespace std;

Scalar activationFunction(Scalar x)
{
    return tanhf(x);
}

Scalar activationFunctionDerivative(Scalar x)
{
    return 1 - tanhf(x) * tanhf(x);
}

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
#ifdef DEBUG
    cout << "Constructor is called!" << endl;
#endif

    this->topology = topology;
    this->learningRate = learningRate;
    for (uint i=0; i<topology.size(); ++i) {
        // initialize neuron layers
        // we add an extra bias neuron to each layer (vector), except for the output one
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i])); // we are using vectors here as layors and not 2d matrices, we are using stohastic gradient descent
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1));

        // initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        // vector.back() gives the handle to recently added element
        // coeffRef gives the reference of value at that place /** Direct access to the underlying index vector */ -> this comes from eigen lib
        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // initialize weights matrix
        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero(); // set all elements to zero in the last column of the weights matrix
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0; // except for the last element -> this is due to the bias neuron
            }
            else {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
};

void NeuralNetwork::propagateForward(RowVector& input)
{
    // set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    // propagate the data forward and then 
    // apply the activation function to your network
    // unaryExpr applies the given function to all elements of CURRENT_LAYER
    for (uint i=1; i<topology.size(); ++i) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(function(activationFunction));
    }
}

void NeuralNetwork::calcErrors(RowVector& output)
{
    // calculate the errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // error calculation of hidden layers is different
    // we will begin by the last hidden layer
    // and we will continue till the first hidden layer
    for (uint i=topology.size() - 2; i>0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights()
{
    // topology.size()-1 = weights.size()
    for (uint i=0; i<topology.size() - 1; ++i) {
        // in this loop we are iterating over the different layers (from first hidden to output layer)
        // if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
        if (i != topology.size() - 2) {
            for (uint c=0; c<weights[i]->cols() - 1; ++c) {
                for (uint r=0; r<weights[i]->rows(); ++r) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else {
            for (uint c = 0; c < weights[i]->cols(); c++) {
                for (uint r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

NeuralNetwork::~NeuralNetwork(void)
{
#ifdef DEBUG
    cout << "Destructor is called!" << endl;
#endif

    for (RowVector* p : neuronLayers) {
        delete p;
    }
    neuronLayers.clear();

    for (RowVector* p : cacheLayers) {
        delete p;
    }
    cacheLayers.clear();

    for (RowVector* p : deltas) {
        delete p;
    }
    deltas.clear();

    for (Matrix* p : weights) {
        delete p;
    }
    weights.clear();
}