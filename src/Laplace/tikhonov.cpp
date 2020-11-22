#include <iostream>
#include <fstream>
#include <ios>

#include <string>
#include <vector>

#include "tikhonov.h"
#include "./include/nmrinv_core.h"

using namespace std;

void read_curve_from_txt_file(string file_path, int skiprows, vector<double> &t2_decay_times, vector<double> &t2_decay_amps)
{
	double value0, value1;
	ifstream in(file_path, std::ios::in);
	int index = 0;
	string crap0, crap1;
	if (in)
	{
		while (index < skiprows)
		{
			in >> crap0 >> crap1;
			index += 1;
		}

		while (in >> value0 >> value1)
		{
			t2_decay_times.push_back(value0);
			t2_decay_amps.push_back(value1);
		}
		return;
	}
	throw(errno);
}

void write_curves_2_txt_file(string file_path, vector<double> &x, vector<double> &y, string x_label, string y_label)
{
	const size_t num_points = x.size();
	ofstream in(file_path, std::ios::out);
	if (in)
	{
		// in << x_label << "\t" << y_label << endl;
		for (int i = 0; i < num_points; i++)
		{
			in << x[i] << "\t" << y[i] << endl;
		}
		return;
	}
	throw(errno);
}

void laplaceInverse(string filePath, string fileName)
{
	string dataFile = filePath + fileName;
	vector<double> decay_times, decay_amps;
	read_curve_from_txt_file(dataFile, 0, decay_times, decay_amps);

	NMRInverterConfig nmr_inv_config(0.1, 1e4, true, 128, -4, 2, 512, 512, 0.0);

	NMRInverter nmr_inverter;
	nmr_inverter.set_config(nmr_inv_config, decay_times);
	// nmr_inverter.set_inversion_lambda(0.15);
	nmr_inverter.find_best_lambda(decay_amps.size(), decay_amps.data());
	nmr_inverter.invert(decay_amps.size(), decay_amps.data());

	string outputFile = filePath + "/NMR_T2.txt";
	write_curves_2_txt_file(outputFile, nmr_inverter.used_t2_bins,
							nmr_inverter.used_t2_amps, "T2 (ms)", "Amplitude");
}