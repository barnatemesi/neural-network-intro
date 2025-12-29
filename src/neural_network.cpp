#include "neural_network.h"

NeuralNetwork::NeuralNetwork(const vector<uint>& topology, const RowVector& inputScaling, Scalar learningRate)
{
#ifdef DEBUG
    cout << "Constructor is called!" << endl;
#endif
    this->topology = topology;
    this->inputScaling = inputScaling;
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
}

RowVector NeuralNetwork::propagateForward(const RowVector& input)
{
    // pre-process inputs
    if (input.size() != inputScaling.size()) {
        cout << "unhandled error has been encountered due to size mismatch!" << endl;
        RowVector zeroReturnVector {{0.0F}};
        return zeroReturnVector;
    }

    RowVector scaled_input(input.size());
    scaled_input = input.array() * inputScaling.array(); // Eigen allows assigning array expressions to matrix / vector variables

    // set the input to input layer
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = scaled_input;

    // propagate the data forward and then 
    // apply the activation function to your network
    // unaryExpr applies the given function to all elements of CURRENT_LAYER
    for (uint i=1; i<topology.size(); ++i) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(function(activationFunction));
    }

    return *neuronLayers.back();
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

void NeuralNetwork::updateWeights(void)
{
    // topology.size()-1 = weights.size()
    for (uint i=0; i<topology.size() - 1; ++i) {
        // in this loop we are iterating over the different layers (from first hidden to output layer)
        // if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // if this layer is not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
        if (i != topology.size() - 2) {
            for (uint c=0; c<weights[i]->cols() - 1; ++c) {
                for (uint r=0; r<weights[i]->rows(); ++r) {
                    weights[i]->coeffRef(r, c) += 
                        learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else { // this is the output layer, no bias neuron
            for (uint c=0; c<weights[i]->cols(); ++c) {
                for (uint r=0; r<weights[i]->rows(); ++r) {
                    weights[i]->coeffRef(r, c) += 
                        learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NeuralNetwork::saveWeights(string filename)
{
    ofstream file1(filename);
    for (Matrix* p : weights) {
        long int rows_of_given_matrix = p->rows();
        long int cols_of_given_matrix = p->cols();
        
        file1 << cols_of_given_matrix << endl;
        file1 << rows_of_given_matrix << endl;

        for (uint i=0; i<cols_of_given_matrix; ++i) {
            for (uint j=0; j<rows_of_given_matrix; ++j) {
                file1 << p->coeff(j, i) << endl;
            }
        }
    }

    file1.close();
}

int NeuralNetwork::loadWeights(string filename)
{
    ifstream file(filename);
    string line;
    long int cols_of_given_matrix = 0;
    long int rows_of_given_matrix = 0;

    for (Matrix* p : weights) { // a check is missing here to check if the structure of the neural network is matching or not
        getline(file, line, '\n');
        cols_of_given_matrix = stoi(line);
        getline(file, line, '\n');
        rows_of_given_matrix = stoi(line);

        if ((cols_of_given_matrix != p->cols()) || (rows_of_given_matrix != p->rows())) {
            return -1; // return -1 error code indicating mismatch in matrix sizes
        }

        for (uint i=0; i<cols_of_given_matrix; ++i) {
            for (uint j=0; j<rows_of_given_matrix; ++j) {
                getline(file, line, '\n');
                p->coeffRef(j, i) = stof(line);
            }
        }
    }

    return 0;
}

void NeuralNetwork::printWeights(void)
{
    for (Matrix* p : weights) {
        cout << *p << endl;
    }
}

void NeuralNetwork::propagateBackward(RowVector& output)
{
    calcErrors(output);
    updateWeights();
}

vector<Scalar> NeuralNetwork::train(vector<RowVector*> input_data, vector<RowVector*> output_data)
{
    vector<Scalar> MS_error;

    for (uint i=0; i<input_data.size(); ++i) {
#ifdef DEBUG_TRAIN
        cout << "Input to neural network is : " << *input_data[i] << endl;
#endif

        (void)propagateForward(*input_data[i]);
#ifdef DEBUG_TRAIN
        cout << "Expected output is : " << *output_data[i] << endl;
        cout << "Output produced is : " << *neuronLayers.back() << endl;
#endif

        propagateBackward(*output_data[i]);
        MS_error.push_back(sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()));
#ifdef DEBUG_TRAIN
        cout << "MSE : " << MS_error.back() << endl;
#endif
    } // end of for loop

    return MS_error;
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

#ifdef ACTIVATION_FN_IS_TANH
Scalar activationFunction(Scalar x)
{
    return tanhf(x);
}
Scalar activationFunctionDerivative(Scalar x)
{
    return 1 - tanhf(x) * tanhf(x);
}
#endif

#ifdef ACTIVATION_FN_IS_SIGMOID
Scalar activationFunction(Scalar x)
{
    return 1 / (1 + exp(-x));
}
Scalar activationFunctionDerivative(Scalar x)
{
    return (activationFunction(x) * (1 - activationFunction(x)));
}
#endif

#ifdef ACTIVATION_FN_IS_RELU
Scalar activationFunction(Scalar x)
{
    if (x < 0.0F) {
        return 0.0F;
    } else {
        return x;
    }
}
Scalar activationFunctionDerivative(Scalar x)
{
    if (x < 0.0F) {
        return 0.0F;
    } else {
        return 1.0F;
    }
}
#endif