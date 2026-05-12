#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <cstdio>
#include "main.h"
#include "neural_network.h"
#include "misc.h"

using namespace std;

// ===== genData =====

TEST_CASE("genData creates input and output files", "[misc]")
{
    genData("test_gendata");

    ifstream f_in("test_gendata-in");
    ifstream f_out("test_gendata-out");

    REQUIRE(f_in.is_open());
    REQUIRE(f_out.is_open());

    // Verify files are non-empty
    string line;
    int count = 0;
    while (getline(f_in, line))
    {
        ++count;
    }
    REQUIRE(count == 1000);

    f_in.close();
    f_out.close();
    remove("test_gendata-in");
    remove("test_gendata-out");
}

// ===== ReadCSV =====

TEST_CASE("ReadCSV returns error for non-existent file", "[misc]")
{
    vector<RowVector *> data;
    int ret = ReadCSV("nonexistent_file.csv", data);
    REQUIRE(ret == -1);
    REQUIRE(data.empty());
}

TEST_CASE("ReadCSV reads CSV data correctly", "[misc]")
{
    // Create a small test CSV
    {
        ofstream f("test_read.csv");
        f << "1.0,2.0,3.0" << endl;
        f << "4.0,5.0,6.0" << endl;
        f << "7.0,8.0,9.0" << endl;
        f.close();
    }

    vector<RowVector *> data;
    int ret = ReadCSV("test_read.csv", data);
    REQUIRE(ret == 0);
    REQUIRE(data.size() == 3);

    REQUIRE(data[0]->size() == 3);
    REQUIRE(data[0]->coeff(0) == 1.0F);
    REQUIRE(data[0]->coeff(1) == 2.0F);
    REQUIRE(data[0]->coeff(2) == 3.0F);

    REQUIRE(data[1]->coeff(0) == 4.0F);
    REQUIRE(data[2]->coeff(2) == 9.0F);

    DeleteData(data);
    remove("test_read.csv");
}

TEST_CASE("ReadCSV reads KF validation data", "[misc]")
{
    vector<RowVector *> data;
    int ret = ReadCSV("data/SIM_KF_validation_inputs.csv", data);
    REQUIRE(ret == 0);
    REQUIRE(data.size() > 0);
    REQUIRE(data[0]->size() == 3);

    DeleteData(data);
}

// ===== WriteCSV =====

TEST_CASE("WriteCSV writes data and can be read back", "[misc]")
{
    vector<Scalar> write_data = {1.5F, 2.5F, 3.5F};
    WriteCSV("test_write.csv", write_data);

    ifstream f("test_write.csv");
    REQUIRE(f.is_open());

    string line;
    vector<Scalar> read_back;
    while (getline(f, line))
    {
        if (!line.empty())
        {
            read_back.push_back(stof(line));
        }
    }
    f.close();

    REQUIRE(read_back.size() == 3);
    REQUIRE(read_back[0] == 1.5F);
    REQUIRE(read_back[1] == 2.5F);
    REQUIRE(read_back[2] == 3.5F);

    remove("test_write.csv");
}

// ===== DeleteData =====

TEST_CASE("DeleteData clears vector and frees memory", "[misc]")
{
    vector<RowVector *> data;
    data.push_back(new RowVector(3));
    data.push_back(new RowVector(3));
    REQUIRE(data.size() == 2);

    DeleteData(data);
    REQUIRE(data.empty());
}
