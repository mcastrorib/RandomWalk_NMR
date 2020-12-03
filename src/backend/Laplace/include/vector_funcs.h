/*

simple functions for vector operations

*/


#ifndef VECTOR_FUNCS_H
#define VECTOR_FUNCS_H

#include <math.h>
#include <vector>
#include <algorithm>

using std::vector;

struct Curve
{
	vector<double> x;
	vector<double> y;
};


vector<double> get_vec_from_ptr(double *ptr, int size);

int get_vector_max_element(const vector<double> &vec);

vector<double> square_vector(const vector<double> &vec);

vector<double> multiply_vector(const vector<double> &vec0, const vector<double> &vec1);
vector<double> multiply_vector(const vector<int> &data, const double value);
vector<double> multiply_vector(const vector<double> &data, const double value);

vector<double> div_vector(const vector<double> &vec0, const vector<double> &vec1);
vector<double> div_vector(const vector<int> &data, const double value);
vector<double> div_vector(const vector<double> &data, const double value);

vector<double> log10_vector(const vector<double> &data);

vector<int> arange_int_vector(const int start, const int end, const int step=1);

vector<double> linspace_double_vector(const double start, const double end, const int size, const bool include_final = true);

vector<double> logspace_vector(const double start, const double end, const int size, const bool include_final = true);

void get_multi_thread_loop_limits(const int tid, const int num_threads, const int num_loops, int &start, int &finish);

#endif