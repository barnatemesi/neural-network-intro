#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <string>
#include "main.h"
#include "misc.h"
#include "neural_network.h"

using namespace std;

// user function prototypes
void do_eigen_lib_test(void);
int do_equation_based_training(void);
int do_kf_based_training(void);
int calculate_outs_based_on_nn(const string& weights_file_name, const string& inputs_csv, const string& output_csv);

#define TOPOLOGY_EQ             {2U, 3U, 1U}
#define TOPOLOGY_KF             {3U, 4U, 1U}

int length_of_training = 10;
Scalar training_rate_inp = 0.005F;

int main(int argc, char *argv[])
{
    std::string input_scaling_vector;
    // handling of input arguments
    if (argc > 1) {
        for (int i=1; i<argc; ++i){
            if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
                cout << "help is triggered!" << endl;
                cout << "-r, --rate: add desired training rate as float!" << endl;
                cout << "-l, --len: add desired length of training!" << endl;
                cout << "-s, --scaling: add desired input scaling vector! must match input vector dimensions! Comma separated list!" << endl;

                return 0;
            }
            if (strcmp(argv[i], "-r")==0 || strcmp(argv[i], "--rate")==0) {
                if ((i+1) < argc) {
                    training_rate_inp = atof(argv[i+1]);
                }
                cout << "chosen training_rate_inp " << training_rate_inp << endl;
            }
            if (strcmp(argv[i], "-l")==0 || strcmp(argv[i], "--len")==0) {
                if ((i+1) < argc) {
                    length_of_training = atoi(argv[i+1]);
                }
                cout << "chosen length_of_training " << length_of_training << endl;
            }
            if (strcmp(argv[i], "-s")==0 || strcmp(argv[i], "--scaling")==0) {
                if ((i+1) < argc) {
                    input_scaling_vector = argv[i+1];
                }
                cout << "chosen input_scaling_vector " << input_scaling_vector << endl;
            }
        }
    }

#ifdef DEBUG
    do_eigen_lib_test();
#endif

#ifdef BASED_EQUATION
    do_equation_based_training();
#endif

#ifdef BASED_KF
    do_kf_based_training();
#endif

#ifdef USE_NN
    calculate_outs_based_on_nn("kf_simple_weights.csv", "data/SIM_KF_validation_inputs.csv", "outputs.csv");
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

    vector<RowVector*> row_vector_test;
    row_vector_test.push_back(new RowVector(1));

    row_vector_test[0]->coeffRef(0) = 5.0F;

    cout << "row_vector_test is: " << row_vector_test[0]->coeff(0) << endl;

    // post-processing
    delete row_vector_test.back();

    cout << "***************" << endl;
}

int do_equation_based_training(void)
{
    vector<RowVector*> in_dat;
    vector<RowVector*> out_dat;

    genData("test");
    int ret = ReadCSV("test-in", in_dat);
    if (ret) {
        cout << "File could not be opened!" << endl;
        return -1;
    }
    ret = ReadCSV("test-out", out_dat);
    if (ret) {
        cout << "File could not be opened!" << endl;
        return -1;
    }

    constexpr uint max_num_of_tries = 5U;
    uint curr_num_of_tries = 0U;
    constexpr Scalar test_val_x = 2.0F;
    constexpr Scalar test_val_y = 3.0F;
    constexpr Scalar expected_output = 2 * test_val_x + 10.0 + test_val_y; // 2 * x + 10 + y
    constexpr Scalar epsilon = 0.5F;
    RowVector run_in_data(2);
    run_in_data << test_val_x, test_val_y;
    RowVector run_out_data;
    run_out_data << 0.0F;
    Scalar sum_of_MS_error = 0.0F;

    while (curr_num_of_tries < max_num_of_tries) {
        RowVector input_scaling_data(2);
        input_scaling_data << 1.0F, 1.0F;
        NeuralNetwork n_network(TOPOLOGY_EQ, input_scaling_data, training_rate_inp);

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

        run_out_data = n_network.propagateForward(run_in_data);

        if (NeuralNetwork::float_cmp(run_out_data(0), expected_output, epsilon)) {
            break;
        } else {
            ++curr_num_of_tries;
        }
    } // end of while loop

    cout << "current number if tries: " << curr_num_of_tries << endl;

    if (NeuralNetwork::float_cmp(run_out_data(0), expected_output, epsilon)) {
        cout << "the test has passed!" << endl;
    } else {
        cout << "expected value is: " << expected_output << endl;
        cout << "the test has failed!" << endl;
    }

    cout << "run_out_data: " << run_out_data(0) << endl;

    // post-processing
    DeleteData(in_dat);
    DeleteData(out_dat);

    cout << "******************************" << endl;

    return 0;
}

int do_kf_based_training(void)
{
    constexpr Scalar ms_error_threshold = 0.900F;

    const Scalar kf_training_rate = 0.005F;
    const int kf_length_of_training = 500;

    vector<RowVector*> in_dat_kf;
    vector<RowVector*> out_dat_kf;
    // RowVector input_scaling_data {{1.0F/60.0F, 1.0F/10.0F, 1.0F/10.0F}};
    RowVector input_scaling_data(3);
    input_scaling_data << 1.0F/10.0F, 1.0F/1.0F, 1.0F/1.0F;
    
    NeuralNetwork n_network_kf(TOPOLOGY_KF, input_scaling_data, kf_training_rate);

    // these inputs / outputs are very simple, the load just goes up to ~ 7Nm through a first-order filter
    // input is such as: omega_shaft, T_mot, T_user
    string input_data_csv = "data/SIM_KF_validation_inputs.csv";
    string output_data_csv = "data/SIM_KF_validation_outputs.csv";
    int ret = ReadCSV(input_data_csv, in_dat_kf);
    if (ret) {
        cout << "File could not be opened! " << input_data_csv << endl;
        return -1;
    }
    // output is: T_load
    ret = ReadCSV(output_data_csv, out_dat_kf);
    if (ret) {
        cout << "File could not be opened! " << output_data_csv << endl;
        return -1;
    }
    
    n_network_kf.printWeights();

    for (int i=0; i<kf_length_of_training; ++i) {
        vector<Scalar> return_val = n_network_kf.train(in_dat_kf, out_dat_kf);
        // cout << "*********************" << endl;
        Scalar sum_of_MS_error = accumulate(return_val.begin(), return_val.end(), 0.0);
        // cout << "sum of all MS error: " << sum_of_MS_error << endl;

        if (sum_of_MS_error < ms_error_threshold) {
            cout << "exited at idx: " << i << endl;
            break;
        }
        if (i == (kf_length_of_training - 1)) {
            cout << "no break / exit condition was triggered" << endl;
        }
    }
    
    cout << "after training *********" << endl;

    n_network_kf.printWeights();

    string kf_weights_file_name = "kf_simple_weights.csv";
    n_network_kf.saveWeights(kf_weights_file_name);

    ret = n_network_kf.loadWeights(kf_weights_file_name);
    if (ret == MISMATCH_IN_SIZE) {
        cout << "Could not load weights! Please check the NN system initialization!" << endl;
        DeleteData(in_dat_kf);
        DeleteData(out_dat_kf);
        return -1;
    }

    cout << "******************************" << endl;
    cout << "test with random sample ******" << endl;

    constexpr Scalar epsilon = 0.99F;
    RowVector run_in_data(3);
    run_in_data << 0.0F, 100.0F, 24.0F;

    Scalar kf_sim_expected_val = run_in_data(1) + run_in_data(2);
    RowVector run_out_data;

    run_out_data = n_network_kf.propagateForward(run_in_data);

    if (NeuralNetwork::float_cmp(run_out_data(0), kf_sim_expected_val, epsilon)) {
        cout << "the test has passed!" << endl;
    } else {
        cout << "expected value is: " << kf_sim_expected_val << endl;
        cout << "the test has failed!" << endl;
        return -1;
    }

    cout << "run_out_data: " << run_out_data(0) << endl;

    // post-processing
    DeleteData(in_dat_kf);
    DeleteData(out_dat_kf);

    cout << "******************************" << endl;

    return 0;
}

int calculate_outs_based_on_nn(const string& weights_file_name, const string& inputs_csv, const string& output_csv)
{
    vector<RowVector*> in_data;
    // we make the simplification that the data is always scalar and of type float
    vector<Scalar> f_out_data;
    // vector<RowVector*> out_data;
    RowVector run_out_data;
    RowVector input_scaling_data(3);
    input_scaling_data << 1.0F, 1.0F, 1.0F;
    NeuralNetwork n_network(TOPOLOGY_KF, input_scaling_data, training_rate_inp);

    int ret = n_network.loadWeights(weights_file_name);
    if (ret == MISMATCH_IN_SIZE) {
        cout << "Could not load weights! Please check the NN system initialization!" << endl;
        return -1;
    }

    n_network.printWeights();

    ret = ReadCSV(inputs_csv, in_data);
    if (ret) {
        cout << "File could not be opened!" << endl;
        return -1;
    }

    // compute outputs based on the input vector
    for (uint i=0; i<in_data.size(); ++i) {
        run_out_data = n_network.propagateForward(*in_data[i]);
        f_out_data.push_back(run_out_data.coeff(0));
    }

    WriteCSV(output_csv, f_out_data);

    // post-processing
    DeleteData(in_data);

    return 0;
}