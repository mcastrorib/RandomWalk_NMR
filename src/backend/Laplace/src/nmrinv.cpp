#include "nmrinv.h"
#include <mutex>

std::mutex mutex;
NMRInverter nmr_inverter;

void initialize_inverter(
	int num_raw_bins, double *raw_bins, double min_t2, double max_t2, 
	int num_t2_bins, int num_lambdas, int prune_num, double noise_amp)
{
	mutex.lock();
	NMRInverterConfig inv_config(min_t2, max_t2, true, num_t2_bins,
		-4.0, 2.0, num_lambdas, prune_num, noise_amp);
	nmr_inverter.set_config(inv_config, num_raw_bins, raw_bins);
	mutex.unlock();
}

void initialize_inverter_with_t2_bins(
	int num_raw_bins, double *raw_bins, int num_t2_bins, 
	double *t2_bins, int num_lambdas, int prune_num, double noise_amp)
{
	mutex.lock();
	NMRInverterConfig inv_config(-1, 4, true, num_t2_bins,
		-4.0, 2.0, num_lambdas, prune_num, noise_amp);
	nmr_inverter.set_config(inv_config, num_raw_bins, raw_bins, num_t2_bins, t2_bins);
	mutex.unlock();
}

double find_best_lambda(int num_raw_amps, double *raw_amps)
{
	mutex.lock();
	nmr_inverter.find_best_lambda(num_raw_amps, raw_amps);
	const double used_tikhonov = nmr_inverter.used_lambda;
	mutex.unlock();
	return used_tikhonov;
}

void get_processed_raw_amps(int data_size, double *raw_amps, double *processed_raw_amps)
{
	mutex.lock();
	nmr_inverter.process_raw_amps(data_size, raw_amps);
	std::copy(nmr_inverter.used_raw_amps.begin(), 
		nmr_inverter.used_raw_amps.end(), processed_raw_amps);
	mutex.unlock();
}

double get_tikhonov_lambda()
{
	mutex.lock();
	double tikhonov_lambda = 0.0;
	if (nmr_inverter.lambda_set) 
	{ 
		tikhonov_lambda = nmr_inverter.used_lambda;
	}
	mutex.unlock();
	return tikhonov_lambda;
}

void set_lambda(double tik_lambda)
{
	mutex.lock();
	nmr_inverter.set_inversion_lambda(tik_lambda);
	mutex.unlock();
}

bool get_noise(double *out_noise_data)
{
	mutex.lock();
	if (!nmr_inverter.inv_config.noise_amp)
	{
		mutex.unlock();
		return false;
	}
	std::copy(nmr_inverter.used_raw_noise.begin(), 
		nmr_inverter.used_raw_noise.end(), out_noise_data);
	mutex.unlock();
	return true;
}

void set_noise(double noise_amp, double* noise_data_ptr)
{
	mutex.lock();
	nmr_inverter.inv_config.noise_amp = noise_amp;
	if (noise_amp)
	{
		nmr_inverter.used_raw_noise.resize(nmr_inverter.orig_num_echos);
		std::copy(noise_data_ptr, noise_data_ptr + nmr_inverter.orig_num_echos, 
			nmr_inverter.used_raw_noise.begin());
	}
	mutex.unlock();
}

void get_used_raw_bins(double *out_used_raw_bins)
{
	mutex.lock();
	std::copy(nmr_inverter.used_raw_bins.begin(),
		nmr_inverter.used_raw_bins.end(), out_used_raw_bins);
	mutex.unlock();
}

void get_used_t2_bins(double *out_used_t2_bins)
{
	mutex.lock();
	std::copy(nmr_inverter.used_t2_bins.begin(),
		nmr_inverter.used_t2_bins.end(), out_used_t2_bins);
	mutex.unlock();
}

bool invert(int data_size, double *raw_amps, double *out_result)
{
	mutex.lock();
	if (!nmr_inverter.lambda_set)
	{
		mutex.unlock();
		return false;
	}
	nmr_inverter.invert(data_size, raw_amps);
	std::copy(nmr_inverter.used_t2_amps.begin(), 
		nmr_inverter.used_t2_amps.end(), out_result);
	mutex.unlock();
	return true;
}

bool invert_with_multiple_noises(const int raw_amps_size, double* raw_amps_ptr,
	const int num_inversions, double* raw_noise_ptr, double* t2_amps_ptr)
{
	mutex.lock();
	if (!nmr_inverter.lambda_set)
	{
		mutex.unlock();
		return false;
	}
	nmr_inverter.invert_with_multiple_noises(raw_amps_size, 
		raw_amps_ptr, num_inversions, raw_noise_ptr, t2_amps_ptr);
	mutex.unlock();
	return true;
}

bool get_lcurve_values(double* solution_norms_ptr, 
	double* residual_norms_ptr, double* lambdas_ptr, double* curvature_ptr)
{
	mutex.lock();
	if (!nmr_inverter.reg_finder.set)
	{
		mutex.unlock();
		return false;
	}
	std::copy(nmr_inverter.reg_finder.solution_norms.begin(),
		nmr_inverter.reg_finder.solution_norms.end(), solution_norms_ptr);
	std::copy(nmr_inverter.reg_finder.residual_norms.begin(),
		nmr_inverter.reg_finder.residual_norms.end(), residual_norms_ptr);
	std::copy(nmr_inverter.reg_finder.lambdas.begin(), 
		nmr_inverter.reg_finder.lambdas.end(), lambdas_ptr);
	std::copy(nmr_inverter.reg_finder.curvature.begin(),
		nmr_inverter.reg_finder.curvature.end(), curvature_ptr);
	mutex.unlock();
	return true;
}
