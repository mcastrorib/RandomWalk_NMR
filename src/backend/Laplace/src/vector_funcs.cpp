#include "vector_funcs.h"

vector<double> get_vec_from_ptr(double* ptr, int size)
{
	vector<double> vec(size);
	std::copy(ptr, ptr + size, vec.data());
	return vec;
}

int get_vector_max_element(const vector<double> &vec)
{
	return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

vector<double> square_vector(const vector<double> &vec)
{
	return multiply_vector(vec, vec);
}

vector<double> multiply_vector(const vector<double> &vec0, const vector<double> &vec1)
{
	vector<double> return_array;
	return_array.reserve(vec0.size());
	for (int i = 0; i < vec0.size(); i++) { return_array.emplace_back(vec0[i] * vec1[i]); }
	return return_array;
}

vector<double> multiply_vector(const vector<int> &data, const double value)
{
	vector<double> return_array;
	return_array.reserve(data.size());
	for (int i = 0; i < data.size(); i++) { return_array.emplace_back((double)data[i] * value); }
	return return_array;
}

vector<double> multiply_vector(const vector<double> &data, const double value)
{
	vector<double> return_array;
	return_array.reserve(data.size());
	for (int i = 0; i < data.size(); i++) { return_array.emplace_back(data[i] * value); }
	return return_array;
}

vector<double> div_vector(const vector<double> &vec0, const vector<double> &vec1)
{
	vector<double> return_array;
	return_array.reserve(vec0.size());
	for (int i = 0; i < vec0.size(); i++) { return_array.emplace_back(vec0[i] / vec1[i]); }
	return return_array;
}

vector<double> div_vector(const vector<int> &data, const double value)
{
	return multiply_vector(data, 1.0 / value);
}

vector<double> div_vector(const vector<double> &data, const double value)
{
	return multiply_vector(data, 1.0 / value);
}

vector<double> log10_vector(const vector<double> &data)
{
	vector<double> return_array;
	return_array.reserve(data.size());
	for (int i = 0; i < data.size(); i++) { return_array.emplace_back(log10(data[i])); }
	return return_array;
}


vector<int> arange_int_vector(const int start, const int end, const int step)
{
	int value = start;
	int size = abs((int)((end - start) / step));
	vector<int> return_array;
	return_array.reserve(size);
	for (int i = 0; i < size; i++)
	{
		return_array.emplace_back(value);
		value += step;
	}
	return return_array;
}

vector<double> linspace_double_vector(const double start, const double end, const int size, const bool include_final)
{
	double step;
	if (include_final) { step = (end - start) / (size - 1); }
	else { step = (end - start) / (size); }

	vector<double> return_array;
	return_array.reserve(size);
	double value = start;
	for (int i = 0; i < size; i++)
	{
		return_array.emplace_back(value);
		value += step;
	}
	return return_array;
}	

vector<double> logspace_vector(const double start, const double end, const int size, const bool include_final)
{
	vector<double> return_array = linspace_double_vector(start, end, size, include_final);
	for (int i = 0; i < return_array.size(); i++) { return_array[i] = pow(10.0, return_array[i]); }
	return return_array;
}

void get_multi_thread_loop_limits(const int tid, const int num_threads, const int num_loops, int &start, int &finish)
{
	const int loops_per_thread = num_loops / num_threads;
	start = tid * loops_per_thread;
	if (tid == (num_threads - 1)) { finish = num_loops; }
	else { finish = start + loops_per_thread; }
}
