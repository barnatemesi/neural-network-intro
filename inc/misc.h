#ifndef MISC_H
#define MISC_H

#include<string>
#include<vector>
#include "main.h"
#include "neural_network.h"

// User function declaration
int ReadCSV(const string& filename, vector<RowVector*>& data);
void DeleteData(vector<RowVector*>& data);
void WriteCSV(const string& filename, const vector<Scalar>& data);
void genData(const string& filename);

#endif