#include "misc.h"

void ReadCSV(string filename, vector<RowVector*>& data)
{
    data.clear();
    ifstream file(filename);
    string line, word;
    // determine number of columns in file
    getline(file, line, '\n');
    stringstream ss(line);
    vector<Scalar> parsed_vec;

    while (getline(ss, word, ',')) {
        parsed_vec.push_back(Scalar(stof(&word[0])));
    }

    uint cols = parsed_vec.size();
    data.push_back(new RowVector(cols));

    for (uint i=0; i<cols; i++) {
        data.back()->coeffRef(1, i) = parsed_vec[i];
    }

    // read the file
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            stringstream ss(line);
            data.push_back(new RowVector(1, cols));
            uint i = 0;
            while (getline(ss, word, ',')) {
                data.back()->coeffRef(i) = Scalar(stof(&word[0]));
                ++i;
            }
        }
    }
}

void genData(string filename)
{
    constexpr uint lenght_of_desired_data = 1000;
    ofstream file1(filename + "-in");
    ofstream file2(filename + "-out");

    for (uint r=0; r<lenght_of_desired_data; r++) {
        Scalar x = rand() / Scalar(RAND_MAX);
        Scalar y = rand() / Scalar(RAND_MAX);

        file1 << x << ", " << y << endl;
        file2 << 2 * x + 10 + y << endl;
    }

    file1.close();
    file2.close();
}

void WriteCSV(string filename, const vector<Scalar>& data)
{
    (void)data;
    ofstream file1(filename);
    for (uint i=0; i<data.size(); ++i) {
        file1 << data[i] <<endl;
    }
    file1.close();
}

bool float_cmp_neural(const Scalar val_in1, const Scalar val_in2, const Scalar threshold_in)
{
    return (abs(val_in1 - val_in2) < threshold_in);
}