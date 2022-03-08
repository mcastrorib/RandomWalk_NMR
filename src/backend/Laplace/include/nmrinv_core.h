/*

Author: Pedro Vianna Mesquita

NMRInverter

Provides ways to invert NMR T2 decays into T2 distributions, offering many options...

The main stars of the show are the struct "NMRInverterConfig" and the class "NMRInverter".

"NMRInverterConfig" contains the parameters/options for the inversion. These are:
	"min_t2" is the minimum time value of the generated T2 distribution. Defaults is 0.1 (100us in milisseconds).
	
	"max_t2" is the maximum time value of the generated T2 distribution. Defaults is 10000.0 (10s in milisseconds).
	
	"t2_use_logspace" is whether to use logarithmic or linear space for generating the T2 valus. Dumb option. Use always "true".
	
	"num_t2_bins" is the number of T2 values for the T2 distribution. Default is 256.
	
	"min_lambda", "max_lambda" and "num_lambdas" are the options for the regularization parameters for using the "TikhonovSolver". Should always be default.
	
	"prune_num" is for pruning-averaging the decays' vectors: 
		If 0 -> no pruning is done. 
		If n > 0 -> n will be the size of the pruned vectors. 
		So basically if you set "prune_num" as 512 (default), the decay curve you enter will be pruned (reduced) to a 512-sized curve.
		This pruning is logarithmic, and usually keeps the first values of the decays intact.
		This is because the pruning only starts after the slope of its "reference logspace" is higher than the slope in the time vector (dt between echos).
		Confusing, I know. Can't explain properly here. Sorry. But it works similarly on WinDxp Software. 
		Anyway, pruning was used before mainly for speeding up processing time, by reducing matrixes sizes. These isn't a problem these days...
		But prune-averaging is also good for averaging noise, reducing it. So, try it out.
	
	"noise_amp" is for creating artificial "white" noise (gaussian noise). Useful when inverting simulated decays.
		The value is actually the standard deviation of the gaussian noise.

"NMRInverter" is the class for inverting NMR T2 decays. Functions for using it are:
	"set_config" -> This is the first function you need to calls. It set most of the things to perform an inversion. Inputs are:
		A "NMRInverterConfig" struct for the configuration.
		A vector "raw_bins", which contain the time values of the decays (echo times).
		A optional vector "forced_t2_bins", for forcibily specifies the values of the T2 distribution times.
			When this options is ignored, the T2 distribution times are generated based on the parameters in "NMRInverterConfig".

		The function can either receives these data values as a std::vector, or as a size and pointer to the data.
	
	"set_inversion_lambda" -> Used to set the value for the Tikhonov Regularization.
		if 0.0 -> Tikhonov is not applied.
		if l > 0.0 -> The value l is used.

	"find_best_lambda" -> Search for the best Tikhonov regularization value using the routines in "TikhonovSolver". 
		It automatically calls "set_inversion_lambda" when done.
		Inputs are the amplitude values of the NMR decay curve, corresponding with the time values in "raw_bins".

	"invert" -> inverts the NMR decay received as input. Parameters are the size of the amplitude data, and a pointer to it.
		The inversion is done using Non-Negative-Least-Squares (NNLS.hpp), using (or not) Tikhonov Regularization.
		Depending on what was set in "noise_amp" and "prune_num", noise is (or isn't) added, and pruning-averaging is (isn't) performed.
		This function can invert one or more at a time. It the last case, the size of the data must be num_of_decays * num_values_in_one_decay.
		This function returns void. The result is actually stored in the member vectors below. Together they for the T2 distriburion curve.
			"used_t2_bins" -> T2 distribution times. 
			"used_t2_amps" -> T2 distribution amplitudes. 

The other functions are for internal operations or other specific things used in some of my softwares.
*/

#ifndef NMRINV_CORE_H
#define NMRINV_CORE_H

#include <tuple>
#include <time.h>
#include <random> 
#include <chrono>
#include "NNLS.hpp"
#include "TikhonovSolver.h"

using std::tuple;
using Eigen::NNLS;

#define DEFAULT_PRUNE_NUM 512
#define DEFAULT_NOISE_AMP 0.002

struct NMRInverterConfig
{
	double min_t2;                //Minimum T2 value for the T2 distribution
	double max_t2;                //Maximum T2 value for the T2 distribution
	bool t2_use_logspace;         //whether to use logarithmic space or linear in the T2 distribution. Stupid option. Should always be true.
	int num_t2_bins;              //Number of T2 values for the T2 distribution
	double min_lambda;            //Min for the regularization value
	double max_lambda;
	int num_lambdas;
	int prune_num;
	double noise_amp;

	NMRInverterConfig(double init_min_t2 = 0.1,
		double init_max_t2 = 10000.0, bool init_t2_use_logspace = true,
		int init_num_t2_bins = 256, double init_min_lambda = DEFAULT_MIN_LAMBDA,
		double init_max_lambda = DEFAULT_MAX_LAMBDA, int init_num_lambdas = DEFAULT_NUM_LAMBDAS,
		int init_prune_num = DEFAULT_PRUNE_NUM, double init_noise_amp = DEFAULT_NOISE_AMP);
};

vector<double> get_normal_distribution(const double loc, const double std, const int size);

MatrixXd create_nmr_inversion_matrix(const vector<double> &raw_bins, const vector<double> &t2_bins, const double tikhonov_lambda);
vector<double> get_t2_bins(const NMRInverterConfig &inv_config);
vector<double> get_raw_amps_from_t2(vector<double> &t2_bins, vector<double> t2_amps, vector<double> raw_bins);

class NMRInverter
{
public:
	bool lambda_set;
	double used_lambda;
	int orig_num_echos;

	TikhonovSolver reg_finder;
	NMRInverterConfig inv_config;

	vector<double> used_t2_bins;
	vector<double> used_t2_amps;
	vector<double> orig_raw_bins;
	vector<double> used_raw_bins;
	vector<double> used_raw_amps;
	vector<double> used_raw_noise;

	MatrixXd pure_nmr_inv_matrix;
	MatrixXd used_nmr_inv_matrix;

	vector<NNLS<MatrixXd>> nnls_holder;
	vector<tuple<int, int, int>> pruning_slices;

	NMRInverter() {};
	~NMRInverter();

	void set_config(NMRInverterConfig init_inv_config, const int raw_bins_size, double* raw_bins_ptr);
	void set_config(NMRInverterConfig init_inv_config, vector<double>& raw_bins);

	void set_config(NMRInverterConfig init_inv_config, const int raw_bins_size, 
		double* raw_bins_ptr, const int forced_t2_bins_size, double* forced_t2_bins_ptr);

	void set_config(NMRInverterConfig init_inv_config, 
		vector<double>& raw_bins, vector<double>& forced_t2_bins);

	void set_pruning();
	void generate_noise();
	void set_inversion_lambda(double tikhonov_lambda = 0.0);
	void find_best_lambda(const int raw_amps_size, double* raw_amps_ptr);
	void prepare_nnls_solvers(const int num_threads, const bool lambda_changed);
	void process_raw_amps(const int raw_amps_size, double* raw_amps_ptr);
	void invert(const int raw_amps_size, double* raw_amps_ptr);
	void invert_with_multiple_noises(const int raw_amps_size, double* raw_amps_ptr,
		const int num_inversions, double* raw_noise_ptr, double* t2_amps_ptr);

	vector<double> get_raw_noise() { return this->used_raw_noise; }
	double get_inversion_lambda();
};



#endif // !SIM_AUX_FUNCS_H
