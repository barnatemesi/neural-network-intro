#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "main.h"
#include "misc.h"
#include "neural_network.h"

#define TOPOLOGY {3U, 4U, 1U}

int main()
{
    const Scalar learning_rate = 0.005F;
    const int training_iterations = 500;
    const Scalar ms_error_threshold = 0.900F;

    // input scaling: omega_shaft, T_mot, T_user
    RowVector input_scaling(3);
    input_scaling << 1.0F / 10.0F, 1.0F, 1.0F;

    NeuralNetwork nn(TOPOLOGY, input_scaling, learning_rate);

    // load training data
    vector<RowVector *> inputs;
    vector<RowVector *> outputs;
    if (ReadCSV("data/SIM_KF_validation_inputs.csv", inputs) ||
        ReadCSV("data/SIM_KF_validation_outputs.csv", outputs))
    {
        cout << "Failed to open training data!" << endl;
        return -1;
    }

    // train
    for (int i = 0; i < training_iterations; ++i)
    {
        vector<Scalar> errors = nn.train(inputs, outputs);
        Scalar total_error = accumulate(errors.begin(), errors.end(), 0.0);
        if (total_error < ms_error_threshold)
        {
            cout << "Converged at iteration " << i << endl;
            break;
        }
    }

    nn.saveWeights("kf_simple_weights.csv");

    // test with a sample input
    RowVector sample(3);
    sample << 0.0F, 100.0F, 24.0F;
    RowVector result = nn.propagateForward(sample);
    cout << "Output: " << result(0) << endl;

    DeleteData(inputs);
    DeleteData(outputs);

    return 0;
}