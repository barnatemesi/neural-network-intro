#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include "main.h"
#include "neural_network.h"
#include "misc.h"

using namespace std;

// ===== Eigen Library Basics =====

TEST_CASE("Eigen matrix basic operations", "[eigen]")
{
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);

    REQUIRE(m(0, 0) == 3.0);
    REQUIRE(m(1, 0) == 2.5);
    REQUIRE(m(0, 1) == -1.0);
    REQUIRE(m(1, 1) == 1.5);
}

TEST_CASE("Eigen RowVector allocation and access", "[eigen]")
{
    vector<RowVector *> row_vector_test;
    row_vector_test.push_back(new RowVector(1));
    row_vector_test[0]->coeffRef(0) = 5.0F;

    REQUIRE(row_vector_test[0]->coeff(0) == 5.0F);

    delete row_vector_test.back();
}

// ===== float_cmp =====

TEST_CASE("float_cmp returns true for values within threshold", "[neural_network]")
{
    REQUIRE(NeuralNetwork::float_cmp(1.0F, 1.0F, 0.1F) == true);
    REQUIRE(NeuralNetwork::float_cmp(1.0F, 1.05F, 0.1F) == true);
    REQUIRE(NeuralNetwork::float_cmp(1.0F, 0.95F, 0.1F) == true);
}

TEST_CASE("float_cmp returns false for values outside threshold", "[neural_network]")
{
    REQUIRE(NeuralNetwork::float_cmp(1.0F, 2.0F, 0.5F) == false);
    REQUIRE(NeuralNetwork::float_cmp(0.0F, 1.0F, 0.5F) == false);
}

// ===== NeuralNetwork Construction =====

TEST_CASE("NeuralNetwork can be constructed with valid topology", "[neural_network]")
{
    RowVector scaling(2);
    scaling << 1.0F, 1.0F;
    REQUIRE_NOTHROW(NeuralNetwork({2U, 3U, 1U}, scaling, 0.005F));
}

TEST_CASE("NeuralNetwork can be constructed with KF topology", "[neural_network]")
{
    RowVector scaling(3);
    scaling << 1.0F, 1.0F, 1.0F;
    REQUIRE_NOTHROW(NeuralNetwork({3U, 4U, 1U}, scaling, 0.005F));
}

// ===== Forward Propagation =====

TEST_CASE("propagateForward returns result with correct dimensions", "[neural_network]")
{
    RowVector scaling(2);
    scaling << 1.0F, 1.0F;
    NeuralNetwork nn({2U, 3U, 1U}, scaling, 0.005F);

    RowVector input(2);
    input << 0.5F, 0.5F;

    RowVector output = nn.propagateForward(input);
    REQUIRE(output.size() == 1);
}

TEST_CASE("propagateForward handles input size mismatch", "[neural_network]")
{
    RowVector scaling(2);
    scaling << 1.0F, 1.0F;
    NeuralNetwork nn({2U, 3U, 1U}, scaling, 0.005F);

    RowVector wrong_input(3);
    wrong_input << 0.5F, 0.5F, 0.5F;

    RowVector output = nn.propagateForward(wrong_input);
    // On mismatch, returns a zero vector of size 1
    REQUIRE(output.size() == 1);
    REQUIRE(output(0) == 0.0F);
}

// ===== Equation-Based Training (2*x + 10 + y) =====

TEST_CASE("Equation-based training converges within allowed tries", "[training]")
{
    vector<RowVector *> in_dat;
    vector<RowVector *> out_dat;

    genData("test");
    int ret = ReadCSV("test-in", in_dat);
    REQUIRE(ret == 0);
    ret = ReadCSV("test-out", out_dat);
    REQUIRE(ret == 0);

    constexpr uint max_num_of_tries = 10U;
    constexpr Scalar test_val_x = 2.0F;
    constexpr Scalar test_val_y = 3.0F;
    constexpr Scalar expected_output = 2 * test_val_x + 10.0F + test_val_y;
    constexpr Scalar epsilon = 0.5F;
    constexpr int length_of_training = 50;
    constexpr Scalar training_rate = 0.005F;

    RowVector run_in_data(2);
    run_in_data << test_val_x, test_val_y;
    RowVector run_out_data;
    bool passed = false;

    for (uint tries = 0; tries < max_num_of_tries; ++tries)
    {
        RowVector input_scaling_data(2);
        input_scaling_data << 1.0F, 1.0F;
        NeuralNetwork nn({2U, 3U, 1U}, input_scaling_data, training_rate);

        for (int i = 0; i < length_of_training; ++i)
        {
            nn.train(in_dat, out_dat);
        }

        run_out_data = nn.propagateForward(run_in_data);

        if (NeuralNetwork::float_cmp(run_out_data(0), expected_output, epsilon))
        {
            passed = true;
            break;
        }
    }

    REQUIRE(passed);

    DeleteData(in_dat);
    DeleteData(out_dat);

    // cleanup generated files
    remove("test-in");
    remove("test-out");
}

// ===== KF-Based Training =====

TEST_CASE("KF-based training reduces MSE below threshold", "[training]")
{
    constexpr Scalar ms_error_threshold = 0.900F;
    constexpr Scalar kf_training_rate = 0.005F;
    constexpr int kf_length_of_training = 2000;
    constexpr uint max_num_of_tries = 10U;

    vector<RowVector *> in_dat_kf;
    vector<RowVector *> out_dat_kf;

    int ret = ReadCSV("data/SIM_KF_validation_inputs.csv", in_dat_kf);
    REQUIRE(ret == 0);
    ret = ReadCSV("data/SIM_KF_validation_outputs.csv", out_dat_kf);
    REQUIRE(ret == 0);

    bool exited_early = false;
    for (uint tries = 0; tries < max_num_of_tries; ++tries)
    {
        RowVector input_scaling_data(3);
        input_scaling_data << 1.0F / 10.0F, 1.0F / 1.0F, 1.0F / 1.0F;
        NeuralNetwork nn({3U, 4U, 1U}, input_scaling_data, kf_training_rate);

        for (int i = 0; i < kf_length_of_training; ++i)
        {
            vector<Scalar> return_val = nn.train(in_dat_kf, out_dat_kf);
            Scalar sum_of_MS_error = accumulate(return_val.begin(), return_val.end(), 0.0);
            if (sum_of_MS_error < ms_error_threshold)
            {
                exited_early = true;
                break;
            }
        }
        if (exited_early)
        {
            break;
        }
    }

    REQUIRE(exited_early);

    DeleteData(in_dat_kf);
    DeleteData(out_dat_kf);
}

TEST_CASE("KF-based NN produces correct output after training", "[training]")
{
    constexpr Scalar ms_error_threshold = 0.900F;
    constexpr Scalar kf_training_rate = 0.005F;
    constexpr int kf_length_of_training = 1000;
    constexpr uint max_num_of_tries = 5U;
    // Use inputs within the training data distribution (T_mot~3-5, T_user~0-7)
    constexpr Scalar epsilon = 1.5F;

    vector<RowVector *> in_dat_kf;
    vector<RowVector *> out_dat_kf;

    int ret = ReadCSV("data/SIM_KF_validation_inputs.csv", in_dat_kf);
    REQUIRE(ret == 0);
    ret = ReadCSV("data/SIM_KF_validation_outputs.csv", out_dat_kf);
    REQUIRE(ret == 0);

    RowVector run_in_data(3);
    run_in_data << 20.0F, 4.0F, 3.0F;
    Scalar kf_sim_expected_val = run_in_data(1) + run_in_data(2);
    RowVector run_out_data;
    bool passed = false;

    for (uint tries = 0; tries < max_num_of_tries; ++tries)
    {
        RowVector input_scaling_data(3);
        input_scaling_data << 1.0F / 10.0F, 1.0F / 1.0F, 1.0F / 1.0F;
        NeuralNetwork nn({3U, 4U, 1U}, input_scaling_data, kf_training_rate);

        for (int i = 0; i < kf_length_of_training; ++i)
        {
            vector<Scalar> return_val = nn.train(in_dat_kf, out_dat_kf);
            Scalar sum_of_MS_error = accumulate(return_val.begin(), return_val.end(), 0.0);
            if (sum_of_MS_error < ms_error_threshold)
            {
                break;
            }
        }

        run_out_data = nn.propagateForward(run_in_data);
        if (NeuralNetwork::float_cmp(run_out_data(0), kf_sim_expected_val, epsilon))
        {
            passed = true;
            break;
        }
    }

    REQUIRE(passed);

    DeleteData(in_dat_kf);
    DeleteData(out_dat_kf);
}

// ===== Save / Load Weights =====

TEST_CASE("saveWeights and loadWeights round-trip preserves weights", "[neural_network]")
{
    constexpr Scalar kf_training_rate = 0.005F;
    RowVector input_scaling_data(3);
    input_scaling_data << 1.0F / 10.0F, 1.0F / 1.0F, 1.0F / 1.0F;

    NeuralNetwork nn({3U, 4U, 1U}, input_scaling_data, kf_training_rate);

    RowVector test_input(3);
    test_input << 1.0F, 2.0F, 3.0F;
    RowVector output_before = nn.propagateForward(test_input);

    string weights_file = "test_weights_roundtrip.csv";
    nn.saveWeights(weights_file);

    // Create a new network and load the saved weights
    NeuralNetwork nn2({3U, 4U, 1U}, input_scaling_data, kf_training_rate);
    int ret = nn2.loadWeights(weights_file);
    REQUIRE(ret == NO_ERROR);

    RowVector output_after = nn2.propagateForward(test_input);

    REQUIRE(NeuralNetwork::float_cmp(output_before(0), output_after(0), 0.001F));

    remove(weights_file.c_str());
}

TEST_CASE("loadWeights returns error for non-existent file", "[neural_network]")
{
    RowVector scaling(3);
    scaling << 1.0F, 1.0F, 1.0F;
    NeuralNetwork nn({3U, 4U, 1U}, scaling, 0.005F);

    int ret = nn.loadWeights("nonexistent_file.csv");
    REQUIRE(ret == MISMATCH_IN_SIZE);
}

TEST_CASE("loadWeights returns error for mismatched topology", "[neural_network]")
{
    RowVector scaling3(3);
    scaling3 << 1.0F, 1.0F, 1.0F;
    NeuralNetwork nn1({3U, 4U, 1U}, scaling3, 0.005F);

    string weights_file = "test_weights_mismatch.csv";
    nn1.saveWeights(weights_file);

    // Different topology
    RowVector scaling2(2);
    scaling2 << 1.0F, 1.0F;
    NeuralNetwork nn2({2U, 3U, 1U}, scaling2, 0.005F);

    int ret = nn2.loadWeights(weights_file);
    REQUIRE(ret == MISMATCH_IN_SIZE);

    remove(weights_file.c_str());
}

// ===== Activation Function =====

TEST_CASE("activationFunction returns expected values", "[neural_network]")
{
    Scalar result_pos = NeuralNetwork::activationFunction(1.0F);
    Scalar result_neg = NeuralNetwork::activationFunction(-1.0F);
    Scalar result_zero = NeuralNetwork::activationFunction(0.0F);

    // Positive input should yield positive (or zero) output for all activation fns
    REQUIRE(result_pos >= 0.0F);
    // Zero input should yield zero for tanh, sigmoid(0)=0.5, relu(0)=0
    REQUIRE(result_zero >= 0.0F);
    // Negative input: relu→0, tanh→negative, sigmoid→<0.5
    REQUIRE(result_neg <= result_pos);
}

TEST_CASE("activationFunctionDerivative at zero is non-negative", "[neural_network]")
{
    Scalar derivative_at_zero = NeuralNetwork::activationFunctionDerivative(0.0F);
    // All supported activation functions have derivative >= 0 at x=0
    // tanh'(0) = 1, sigmoid'(0) = 0.25, relu'(0) = 1
    REQUIRE(derivative_at_zero > 0.0F);
}
