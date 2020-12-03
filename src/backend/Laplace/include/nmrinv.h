#ifndef NMRINV
#define NMRINV

#ifdef __linux__
	#define NMRINV_CALLS_API
#elif NMRINV_EXPORTS
	#define NMRINV_CALLS_API __declspec(dllexport)
#else
	#define NMRINV_CALLS_API __declspec(dllimport)
#endif

#include "nmrinv_core.h"

extern "C" NMRINV_CALLS_API void initialize_inverter(
	int num_raw_bins, double *raw_bins, double min_t2, double max_t2, 
	int num_t2_bins, int num_lambdas, int prune_num, double noise_amp);

extern "C" NMRINV_CALLS_API void initialize_inverter_with_t2_bins(
	int num_raw_bins, double *raw_bins, int num_t2_bins, double *t2_bins, 
	int num_lambdas, int prune_num, double noise_amp);

extern "C" NMRINV_CALLS_API double find_best_lambda(
	int num_raw_amps, double *raw_amps);

extern "C" NMRINV_CALLS_API void get_processed_raw_amps(
	int data_size, double *raw_amps, double *processed_raw_amps);

extern "C" NMRINV_CALLS_API bool invert(
	int data_size, double *raw_amps, double *out_result);

extern "C" NMRINV_CALLS_API bool invert_with_multiple_noises(
	const int raw_amps_size, double* raw_amps_ptr,
	const int num_inversions, double* raw_noise_ptr, double* t2_amps_ptr);

extern "C" NMRINV_CALLS_API bool get_lcurve_values(double* solution_norms_ptr,
	double* residual_norms_ptr, double* lambdas_ptr, double* curvature_ptr);

extern "C" NMRINV_CALLS_API double get_tikhonov_lambda();
extern "C" NMRINV_CALLS_API void set_lambda(double tik_lambda);
extern "C" NMRINV_CALLS_API bool get_noise(double *out_noise_data);
extern "C" NMRINV_CALLS_API void set_noise(double noise_amp, double* in_noise_data);
extern "C" NMRINV_CALLS_API void get_used_raw_bins(double *out_used_raw_bins);
extern "C" NMRINV_CALLS_API void get_used_t2_bins(double *out_used_t2_bins);

#endif