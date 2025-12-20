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
    std::vector<RowVector*> in_dat;
    std::vector<RowVector*> out_dat;
    
    return 0;
}