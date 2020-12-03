#ifndef TIKHONOV_H
#define TIKHONOV_H

#include <string>
#include <vector>

using namespace std;

void read_curve_from_txt_file(string file_path, int skiprows, vector<double> &t2_decay_times, vector<double> &t2_decay_amps);
void write_curves_2_txt_file(string file_path, vector<double> &x, vector<double> &y, string x_label, string y_label);
void laplaceInverse(string filePath, string fileName);

#endif