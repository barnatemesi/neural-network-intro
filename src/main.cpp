#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "main.h"
#include "neural_network.h"

using namespace std;

int main()
{
    // eigen lib test-call
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    cout << m << endl;

    // body of algorithm
    NeuralNetwork n({ 2, 3, 1 });
    vector<RowVector*> in_dat;
    vector<RowVector*> out_dat;

    genData("test");
    ReadCSV("test-in", in_dat);
    ReadCSV("test-out", out_dat);
    vector<Scalar> return_val = n.train(in_dat, out_dat);
    
    return 0;
}