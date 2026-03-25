#include "misc.h"
#include <random>

int ReadCSV(const string &filename, vector<RowVector *> &data)
{
	data.clear();
	ifstream file(filename);
	if (!file.is_open())
	{
		return -1;
	}
	string line, word;
	// determine number of columns in file
	getline(file, line, '\n');
	stringstream ss(line);
	vector<Scalar> parsed_vec;

	while (getline(ss, word, ','))
	{
		parsed_vec.push_back(Scalar(stof(&word[0])));
	}

	uint cols = parsed_vec.size();
	data.push_back(new RowVector(cols));

	for (uint i = 0; i < cols; i++)
	{
		data.back()->coeffRef(0, i) = parsed_vec[i];
	}

	// read the file
	if (file.is_open())
	{
		while (getline(file, line, '\n'))
		{
			stringstream ss(line);
			data.push_back(new RowVector(1, cols));
			uint i = 0;
			while (getline(ss, word, ','))
			{
				data.back()->coeffRef(i) = Scalar(stof(&word[0]));
				++i;
			}
		}
	}

	return 0;
}

void DeleteData(vector<RowVector *> &data)
{
	for (RowVector *p : data)
	{
		delete p;
	}
	data.clear();
}

void genData(const string &filename)
{
	constexpr uint lenght_of_desired_data = 1000;
	ofstream file1(filename + "-in");
	ofstream file2(filename + "-out");

	static mt19937 rng(random_device{}());
	uniform_real_distribution<Scalar> dist(0.0F, 1.0F);

	for (uint r = 0; r < lenght_of_desired_data; r++)
	{
		Scalar x = dist(rng);
		Scalar y = dist(rng);

		file1 << x << ", " << y << endl;
		file2 << 2 * x + 10 + y << endl;
	}

	file1.close();
	file2.close();
}

void WriteCSV(const string &filename, const vector<Scalar> &data)
{
	ofstream file1(filename);
	for (uint i = 0; i < data.size(); ++i)
	{
		file1 << data[i] << endl;
	}
	file1.close();
}