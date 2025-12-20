#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include "main.h"
#include "neural_network.h"

using namespace std;

int main(int argc, char *argv[])
{
    int length_of_training = 10U;
    Scalar training_rate_inp = 0.005F;
    if (argc > 3) {
        cout << "too many arguments were passed!" << endl;
    } else if (argc == 3) {
        training_rate_inp = atof(argv[2]);
        length_of_training = atoi(argv[1]);
    } else if (argc == 2) {
        length_of_training = atoi(argv[1]);
    }

    // eigen lib test-call
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    cout << m << endl;
    cout << "***************" << endl;

    // body of algorithm
    NeuralNetwork n({ 2, 3, 1 }, training_rate_inp);
    vector<RowVector*> in_dat;
    vector<RowVector*> out_dat;

    genData("test");
    ReadCSV("test-in", in_dat);
    ReadCSV("test-out", out_dat);

    n.printWeights();
    for (int i=0; i<length_of_training; ++i) {
        vector<Scalar> return_val = n.train(in_dat, out_dat);
        cout << "*********************" << endl;
        cout << "sum of all MS error: " << accumulate(return_val.begin(), return_val.end(), 0.0) << endl;
    }

    cout << "after training *********" << endl;
    n.printWeights();

    return 0;
}