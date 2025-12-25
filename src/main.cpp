#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include "main.h"
#include "neural_network.h"

using namespace std;

void do_eigen_lib_test(void);
void do_equation_based_training(void);
void do_kf_based_training(void);

int length_of_training = 10;
Scalar training_rate_inp = 0.005F;

int main(int argc, char *argv[])
{
    // pre-processing
    if (argc > 3) {
        cout << "too many arguments were passed!" << endl;
    } else if (argc == 3) {
        training_rate_inp = atof(argv[2]);
        length_of_training = atoi(argv[1]);
    } else if (argc == 2) {
        length_of_training = atoi(argv[1]);
    }

    //
    do_eigen_lib_test();

    //
#ifdef BASED_EQUATION
    do_equation_based_training();
#endif

    //
#ifdef BASED_KF
    do_kf_based_training();
#endif

    cout << "end of program ***********" << endl;

    return 0;
}

void do_eigen_lib_test(void)
{
    // eigen lib test-call
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    cout << m << endl;
    cout << "***************" << endl;
}

void do_equation_based_training(void)
{
    vector<RowVector*> in_dat;
    vector<RowVector*> out_dat;

    genData("test");
    ReadCSV("test-in", in_dat);
    ReadCSV("test-out", out_dat);

    constexpr uint max_num_of_tries = 5U;
    uint curr_num_of_tries = 0U;
    constexpr Scalar test_val_x = 2.0F;
    constexpr Scalar test_val_y = 3.0F;
    constexpr Scalar expected_output = 2 * test_val_x + 10.0 + test_val_y; // 2 * x + 10 + y
    constexpr Scalar epsilon = 0.5F;
    RowVector run_in_data {{test_val_x, test_val_y}};
    RowVector run_out_data {{0.0F}};
    Scalar sum_of_MS_error = 0.0F;

    while (curr_num_of_tries < max_num_of_tries) {
        NeuralNetwork n_network({ 2, 3, 1 }, training_rate_inp);

        n_network.printWeights();

        for (int i=0; i<length_of_training; ++i) {
            vector<Scalar> return_val = n_network.train(in_dat, out_dat);
            cout << "*********************" << endl;
            sum_of_MS_error = accumulate(return_val.begin(), return_val.end(), 0.0);
            cout << "sum of all MS error: " << sum_of_MS_error << endl;
        }

        cout << "after training *********" << endl;
        n_network.printWeights();

        cout << "******************************" << endl;
        cout << "test with random sample ******" << endl;

        run_out_data= n_network.propagateForward(run_in_data);

        if (float_cmp_neural(run_out_data(0), expected_output, epsilon)) {
            break;
        } else {
            ++curr_num_of_tries;
        }
    } // end of while loop

    cout << "current number if tries: " << curr_num_of_tries << endl;

    if (float_cmp_neural(run_out_data(0), expected_output, epsilon)) {
        cout << "the test has passed!" << endl;
    } else {
        cout << "expected value is: " << expected_output << endl;
        cout << "the test has failed!" << endl;
    }

    cout << "run_out_data: " << run_out_data(0) << endl;
    cout << "******************************" << endl;
}

void do_kf_based_training(void)
{
    constexpr Scalar ms_error_threshold = 0.900F;

    training_rate_inp = 0.005F;
    length_of_training = 250;

    vector<RowVector*> in_dat_kf;
    vector<RowVector*> out_dat_kf;
    NeuralNetwork n_network_kf({ 3, 4, 1 }, training_rate_inp);

    // these inputs / outputs are very simple, the load just goes up to ~ 7Nm through a first-order filter
    // input is such as: omega_shaft, T_mot, T_user
    ReadCSV("kf-data/SIM_KF_validation_inputs.csv", in_dat_kf);
    // output is: T_load
    ReadCSV("kf-data/SIM_KF_validation_outputs.csv", out_dat_kf);
    
    n_network_kf.printWeights();
    for (int i=0; i<length_of_training; ++i) {
        vector<Scalar> return_val = n_network_kf.train(in_dat_kf, out_dat_kf);
        cout << "*********************" << endl;
        Scalar sum_of_MS_error = accumulate(return_val.begin(), return_val.end(), 0.0);
        cout << "sum of all MS error: " << sum_of_MS_error << endl;

        if (sum_of_MS_error < ms_error_threshold) {
            cout << "exited at idx: " << i << endl;
            break;
        }
        if (i == (length_of_training - 1)) {
            cout << "no break / exit condition was triggered" << endl;
        }
    }
    
    cout << "after training *********" << endl;
    n_network_kf.printWeights();

    string kf_weights_file_name = "kf_simple_weights.csv";
    n_network_kf.saveWeights(kf_weights_file_name);
    n_network_kf.loadWeights(kf_weights_file_name);

    cout << "******************************" << endl;
    cout << "test with random sample ******" << endl;

    constexpr Scalar epsilon = 0.5F;
    RowVector run_in_data {{0.0F, 100.0F, 24.0F}};
    Scalar kf_sim_expected_val = run_in_data(1) + run_in_data(2);
    RowVector run_out_data {{0.0F}};

    run_out_data= n_network_kf.propagateForward(run_in_data);

    if (float_cmp_neural(run_out_data(0), kf_sim_expected_val, epsilon)) {
        cout << "the test has passed!" << endl;
    } else {
        cout << "expected value is: " << kf_sim_expected_val << endl;
        cout << "the test has failed!" << endl;
    }

    cout << "run_out_data: " << run_out_data(0) << endl;

    cout << "******************************" << endl;
}