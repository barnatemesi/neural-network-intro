#include "main.h"
#include<iostream>
#include <Eigen/Dense>

using namespace std;

int main()
{
    cout<<"Welcome to Online IDE!! Happy Coding :)" << endl;

    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    cout << m << endl;
    
    return 0;
}