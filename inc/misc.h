#ifndef MISC_H
#define MISC_H

#include<string>
#include<vector>
#include "main.h"
#include "neural_network.h"

// User function declaration
int ReadCSV(string filename, vector<RowVector*>& data);
void DeleteData(vector<RowVector*>& data);
void WriteCSV(string filename, const vector<Scalar>& data);
void genData(string filename);

#endif