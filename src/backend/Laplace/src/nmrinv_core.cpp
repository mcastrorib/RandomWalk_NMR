#include "nmrinv_core.h"


vector<double> get_normal_distribution(const double loc, const double std, const int size)
{
	std::random_device rd;
    unsigned seed;

    // check if the implementation provides a usable random_device
    if (0 != rd.entropy())
    {
       seed = rd();
    }
    else
    {
       // no random_device available, seed using the system clock
       seed = static_cast<unsigned> (std::chrono::system_clock::now().time_since_epoch().count());
    }

	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(loc, std);
	vector<double> normal_dist;
	normal_dist.reserve(size);
	for (int i = 0; i < size; i++)
	{
		normal_dist.emplace_back(distribution(generator));
	}
	return normal_dist;
}


NMRInverterConfig::NMRInverterConfig(double init_min_t2, double init_max_t2,
	bool init_t2_use_logspace, int init_num_t2_bins, double init_min_lambda,
	double init_max_lambda, int init_num_lambdas, int init_prune_num, double init_noise_amp)
{
	this->min_t2 = init_min_t2;
	this->max_t2 = init_max_t2;
	this->t2_use_logspace = init_t2_use_logspace;
	this->num_t2_bins = init_num_t2_bins;

	this->min_lambda = init_min_lambda;
	this->max_lambda = init_max_lambda;
	this->num_lambdas = init_num_lambdas;

	this->prune_num = init_prune_num;
	this->noise_amp = init_noise_amp;
}

vector<double> get_raw_amps_from_t2(vector<double> &t2_bins, vector<double> t2_amps, vector<double> raw_bins)
{
	const int num_cpu_threads = omp_get_max_threads();
	const int num_raw_bins = raw_bins.size();
	const int num_t2_bins = t2_bins.size();
	vector<double> raw_amps(num_raw_bins);

	#pragma omp parallel num_threads(num_cpu_threads)
	{
		const int tid = omp_get_thread_num(); 
		int i_start, i_finish; double mt, amp_value;
		get_multi_thread_loop_limits(tid, num_cpu_threads, num_raw_bins, i_start, i_finish);
		
		for (int i = i_start; i < i_finish; i++)
		{
			amp_value = 0.0;
			mt = -raw_bins[i];
			for (int n = 0; n < num_t2_bins; n++)
			{
				amp_value += t2_amps[n] * exp(mt / t2_bins[n]);
			}
			raw_amps[i] = amp_value;
		}
	}
	return raw_amps;
}

static void add_noise_to_decays(const int num_decays, const int num_echos, 
	const double* raw_amps, const double* raw_noise, vector<double>& out_raw_amps)
{
	const int num_cpu_threads = omp_get_max_threads();
	#pragma omp parallel num_threads(num_cpu_threads)
	{
		int decay_start_index, index;
		int echo_start, echo_finish;
		const int tid = omp_get_thread_num();
		get_multi_thread_loop_limits(tid, num_cpu_threads, num_echos, echo_start, echo_finish);

		for (int idecay = 0; idecay < num_decays; idecay++)
		{
			decay_start_index = idecay * num_echos;
			for (int iecho = echo_start; iecho < echo_finish; iecho++)
			{
				index = decay_start_index + iecho;
				out_raw_amps[index] = raw_amps[index] + raw_noise[iecho];
			}
		}
	}
}

MatrixXd create_nmr_inversion_matrix(const vector<double>& raw_bins,
 	const vector<double>& t2_bins, const double tikhonov_lambda)
{
	const int M = raw_bins.size();
	const int N = t2_bins.size();

	MatrixXd nmr_inv_matrix;
	if (tikhonov_lambda) { nmr_inv_matrix.resize((M + N), N); }
	else { nmr_inv_matrix.resize(M, N); }
	
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			nmr_inv_matrix(i, j) = exp(-raw_bins[i] / t2_bins[j]);
		}
	}

	if (tikhonov_lambda)
	{
		for (int j = 0; j < N; j++)
		{
			nmr_inv_matrix(M + j, j) = tikhonov_lambda;
		}
	}
	
	return nmr_inv_matrix;
}


vector<double> get_t2_bins(const NMRInverterConfig &inv_config)
{
	if (inv_config.t2_use_logspace)
	{
		return logspace_vector(log10(inv_config.min_t2), 
			log10(inv_config.max_t2), inv_config.num_t2_bins);
	}
	else
	{
		return linspace_double_vector(inv_config.min_t2,
			inv_config.max_t2, inv_config.num_t2_bins);
	}
}


//////////////////////////////////////////////////////// NEW PRUNING
vector<tuple<int, int, int>> get_nmr_pruning_slices(vector<double>& raw_bins, const int num_pruning)
{
	const int num_bins = raw_bins.size();

	vector<double> input_log_bins;
	input_log_bins.reserve(num_bins);
	for (int i = 0; i < num_bins; i++) { input_log_bins.emplace_back(log10(raw_bins[i])); }

	const double min_log_raw_bins = input_log_bins[0];
	double log_bins_diff = input_log_bins[num_bins - 1] - min_log_raw_bins;
	log_bins_diff /= (num_pruning - 1);

	vector<double> new_log_bins;
	new_log_bins.reserve(num_pruning);
	for (int i = 0; i < num_pruning; i++) { new_log_bins.emplace_back(min_log_raw_bins + (log_bins_diff * i)); }

	int p_start_index = 1;
	while ((pow(10.0, new_log_bins[p_start_index]) - raw_bins[p_start_index]) <= 0.0) { p_start_index += 1; }
	p_start_index -= 1;

	int ib = 0;
	double pruning_start_log_bin = new_log_bins[p_start_index];
	while ((input_log_bins[ib] - pruning_start_log_bin) <= 0.0) { ib += 1; }

	double bi, bf; int bin_start_index;
	vector<tuple<int, int, int>> pruning_slices;
	for (int pi = p_start_index; pi < (num_pruning - 1); pi++)
	{
		bin_start_index = ib; bi = new_log_bins[pi];  bf = new_log_bins[pi + 1];
		while ((ib < num_bins) && (input_log_bins[ib] >= bi) && (input_log_bins[ib] < bf)) { ib += 1; }
		pruning_slices.push_back(std::make_tuple(pi, bin_start_index, ib));
	}

	return pruning_slices;
}


void prune_nmr_data_from_slices(
	const int raw_data_size, double* raw_data, const bool use_log, const int num_pruning,
	const vector<tuple<int, int, int>>& pruning_slices, vector<double>& pruned_data)
{
	int num_curves = 1; 
	int num_threads_to_use = 1;
	int num_raw_bins = raw_data_size;
	const int pruning_start_index = std::get<0>(pruning_slices[0]);

	if (pruned_data.size() > num_pruning) 
	{ 
		num_curves = pruned_data.size() / num_pruning; 
		num_raw_bins = raw_data_size / num_curves; 
		num_threads_to_use = omp_get_max_threads();
	}

	#pragma omp parallel num_threads(num_threads_to_use)
	{
		double mean_sum;
		double* curve_raw_data;
		double* curve_pruned_data;
		int p_index, pb_start, pb_end;
		const int tid = omp_get_thread_num();	

		for (int ic = tid; ic < num_curves; ic += num_threads_to_use)
		{
			curve_raw_data = raw_data + (ic * num_raw_bins);
			curve_pruned_data = pruned_data.data() + (ic * num_pruning);
			std::copy(curve_raw_data, curve_raw_data + pruning_start_index, curve_pruned_data);

			for (int si = 0; si < pruning_slices.size(); si++)
			{
				mean_sum = 0.0;
				p_index = std::get<0>(pruning_slices[si]);
				pb_start = std::get<1>(pruning_slices[si]);
				pb_end = std::get<2>(pruning_slices[si]);

				if (use_log)
				{
					for (int pbi = pb_start; pbi < pb_end; pbi++)
					{
						mean_sum += log10(curve_raw_data[pbi]);
					}
					curve_pruned_data[p_index] = pow(10.0, mean_sum / (pb_end - pb_start));
				}
				else
				{
					for (int pbi = pb_start; pbi < pb_end; pbi++)
					{
						mean_sum += curve_raw_data[pbi];
					}
					curve_pruned_data[p_index] = mean_sum / (pb_end - pb_start);
				}
			}
			curve_pruned_data[num_pruning - 1] = curve_raw_data[num_raw_bins - 1];
		}
	}
}



NMRInverter::~NMRInverter()
{
	this->nnls_holder.clear();
}

void NMRInverter::set_config(NMRInverterConfig init_inv_config, const int raw_bins_size, double* raw_bins_ptr)
{
	this->lambda_set = false;
	this->inv_config = init_inv_config;
	this->orig_num_echos = raw_bins_size;
	this->used_t2_bins = get_t2_bins(this->inv_config);
	this->orig_raw_bins.assign(raw_bins_ptr, raw_bins_ptr + raw_bins_size);
	this->generate_noise();
	this->set_pruning();
	this->pure_nmr_inv_matrix = create_nmr_inversion_matrix(this->used_raw_bins, this->used_t2_bins, 0.0);
}

void NMRInverter::set_config(NMRInverterConfig init_inv_config, vector<double>& raw_bins)
{
	this->set_config(init_inv_config, raw_bins.size(), raw_bins.data());
}

void NMRInverter::set_config(NMRInverterConfig init_inv_config, const int raw_bins_size, 
	double* raw_bins_ptr, const int forced_t2_bins_size, double* forced_t2_bins_ptr)
{
	this->lambda_set = false;
	this->inv_config = init_inv_config;
	this->orig_num_echos = raw_bins_size;
	this->orig_raw_bins.assign(raw_bins_ptr, raw_bins_ptr + raw_bins_size);
	this->used_t2_bins.assign(forced_t2_bins_ptr, forced_t2_bins_ptr + forced_t2_bins_size);
	this->generate_noise();
	this->set_pruning();
	this->pure_nmr_inv_matrix = create_nmr_inversion_matrix(this->used_raw_bins, this->used_t2_bins, 0.0);
}


void NMRInverter::set_config(NMRInverterConfig init_inv_config,
	vector<double>& raw_bins, vector<double>& forced_t2_bins)
{
	this->set_config(init_inv_config, raw_bins.size(), 
		raw_bins.data(), forced_t2_bins.size(), forced_t2_bins.data());
}

void NMRInverter::set_pruning()
{
	if (this->inv_config.prune_num)
	{
		this->used_raw_bins.resize(this->inv_config.prune_num);
		this->pruning_slices = get_nmr_pruning_slices(this->orig_raw_bins, this->inv_config.prune_num);
		prune_nmr_data_from_slices(this->orig_raw_bins.size(), this->orig_raw_bins.data(),
			true, this->inv_config.prune_num, this->pruning_slices, this->used_raw_bins);
	}
	else
	{
		this->pruning_slices.clear();
		this->used_raw_bins.resize(this->orig_raw_bins.size());
		std::copy(this->orig_raw_bins.begin(), this->orig_raw_bins.end(), this->used_raw_bins.begin());
	}
}

void NMRInverter::generate_noise()
{
	if (this->inv_config.noise_amp)
	{
		this->used_raw_noise = get_normal_distribution(0.0, 
			this->inv_config.noise_amp, this->orig_num_echos);
	}
	else
	{
		this->used_raw_noise.clear();
	}
}


void NMRInverter::set_inversion_lambda(double tikhonov_lambda)
{
	if (tikhonov_lambda)
	{
		this->used_nmr_inv_matrix.resize(this->used_raw_bins.size() + this->used_t2_bins.size(), this->used_t2_bins.size());
		this->used_nmr_inv_matrix << this->pure_nmr_inv_matrix, MatrixXd::Zero(this->used_t2_bins.size(), this->used_t2_bins.size());
		for (int i = 0; i < this->used_t2_bins.size(); i++)
		{ 
			this->used_nmr_inv_matrix(this->used_raw_bins.size() + i, i) = tikhonov_lambda; 
		}
	}
	else
	{
		this->used_nmr_inv_matrix.resize(this->pure_nmr_inv_matrix.rows(), this->pure_nmr_inv_matrix.cols());
		this->used_nmr_inv_matrix = this->pure_nmr_inv_matrix.replicate(1, 1);
	}
	this->used_lambda = tikhonov_lambda;
	this->lambda_set = true;
	this->prepare_nnls_solvers(1, true);
}

double NMRInverter::get_inversion_lambda()
{
	if(this->lambda_set) return this->used_lambda;
	else return 0.0;
}


void NMRInverter::prepare_nnls_solvers(const int num_threads, const bool lambda_changed)
{
	if ((this->nnls_holder.size() != num_threads) || lambda_changed)
	{
		nnls_holder.clear();
		for (int tid = 0; tid < num_threads; tid++)
		{
			this->nnls_holder.push_back(NNLS<MatrixXd>(this->used_nmr_inv_matrix));
		}
	}
}

void NMRInverter::process_raw_amps(const int raw_amps_size, double* raw_amps_ptr)
{
	const int num_decays = raw_amps_size / this->orig_num_echos;
	const int used_raw_amps_size = num_decays * this->used_raw_bins.size();
	if (this->used_raw_amps.size() != used_raw_amps_size) 
	{ 
		this->used_raw_amps.resize(used_raw_amps_size); 
	}

	if (this->inv_config.prune_num > 0)
	{
		if (this->inv_config.noise_amp)
		{
			vector<double> noisy_raw_amps(raw_amps_size);
			add_noise_to_decays(num_decays, this->orig_num_echos, 
				raw_amps_ptr, this->used_raw_noise.data(), noisy_raw_amps);
			prune_nmr_data_from_slices(raw_amps_size, noisy_raw_amps.data(), false, 
				this->inv_config.prune_num, this->pruning_slices, this->used_raw_amps);
		}
		else 
		{ 
			prune_nmr_data_from_slices(raw_amps_size, raw_amps_ptr, false,
				this->inv_config.prune_num, this->pruning_slices, this->used_raw_amps);
		}
	}
	else
	{
		if (this->inv_config.noise_amp) 
		{ 
			add_noise_to_decays(num_decays, this->orig_num_echos,
				raw_amps_ptr, this->used_raw_noise.data(), this->used_raw_amps);			
		}
		else { std::copy(raw_amps_ptr, raw_amps_ptr + raw_amps_size, this->used_raw_amps.begin()); }
	}
}

void NMRInverter::find_best_lambda(const int raw_amps_size, double* raw_amps_ptr)
{
	this->process_raw_amps(raw_amps_size, raw_amps_ptr);
	this->reg_finder.find_best_lambda(this->pure_nmr_inv_matrix, this->used_raw_amps,
		this->inv_config.min_lambda, this->inv_config.max_lambda, this->inv_config.num_lambdas);
	this->set_inversion_lambda(this->reg_finder.best_lambda);
}


void NMRInverter::invert(const int raw_amps_size, double* raw_amps_ptr)
{
	if (!this->lambda_set)
	{
		cout << endl << "Tikhonov Lambda was not set yet. Aborting inversion..." << endl << endl;
		throw 20;
	}

	clock_t inv_ct = clock();
	this->process_raw_amps(raw_amps_size, raw_amps_ptr);
	const int num_decays = raw_amps_size / this->orig_num_echos;
	if (this->used_t2_amps.size() != (num_decays * this->used_t2_bins.size()))
	{ 
		this->used_t2_amps.resize(num_decays * this->used_t2_bins.size());
	}

	int B_size = this->used_raw_bins.size();
	if (this->used_lambda != 0.0) { B_size += this->used_t2_bins.size(); }

	const int num_cpu_threads = omp_get_max_threads();

	if (num_decays > num_cpu_threads)
	{
		this->prepare_nnls_solvers(num_cpu_threads, false);
		#pragma omp parallel num_threads(num_cpu_threads)
		{
			VectorXd t2_amps;
			VectorXd raw_amps_B(B_size);
			raw_amps_B.setZero();

			int decay_start_index;
			const int tid = omp_get_thread_num();
			for (int idecay = tid; idecay < num_decays; idecay += num_cpu_threads)
			{
				decay_start_index = this->used_raw_bins.size() * idecay;
				std::copy(this->used_raw_amps.begin() + decay_start_index,
					this->used_raw_amps.begin() + decay_start_index + this->used_raw_bins.size(), raw_amps_B.data());

				if (this->nnls_holder[tid].solve(raw_amps_B))
				{
					t2_amps = this->nnls_holder[tid].x();
					std::copy(t2_amps.data(), t2_amps.data() + t2_amps.size(), 
						this->used_t2_amps.begin() + (idecay * this->used_t2_bins.size()));
				}
				else
				{
					cout << endl << "Fuck. Something happened while inverting." << endl << endl;
					throw 20;
				}
			}
		}
	}

	else
	{
		VectorXd t2_amps;
		VectorXd raw_amps_B(B_size);
		raw_amps_B.setZero();

		int decay_start_index;
		for (int idecay = 0; idecay < num_decays; idecay++)
		{
			decay_start_index = this->used_raw_bins.size() * idecay;
			std::copy(this->used_raw_amps.begin() + decay_start_index,
				this->used_raw_amps.begin() + decay_start_index + this->used_raw_bins.size(), raw_amps_B.data());

			if (this->nnls_holder[0].solve(raw_amps_B))
			{
				t2_amps = this->nnls_holder[0].x();
				std::copy(t2_amps.data(), t2_amps.data() + t2_amps.size(), this->used_t2_amps.begin() + (idecay * this->used_t2_bins.size()));
			}
			else
			{
				cout << endl << "Inversion procedure failed." << endl << endl;
				throw 20;
			}
		}
	}
}


void NMRInverter::invert_with_multiple_noises(const int raw_amps_size, double* raw_amps_ptr,
	const int num_inversions, double* raw_noise_ptr, double* t2_amps_ptr)
{
	const int num_decays = raw_amps_size / this->orig_num_echos;
	const int num_total_t2_bins = num_decays * this->inv_config.num_t2_bins;

	vector<double> prev_noise_amps;
	const double prev_noise_std = this->inv_config.noise_amp;
	if (prev_noise_std) 
	{ 
		prev_noise_amps.assign(this->used_raw_noise.begin(), this->used_raw_noise.end()); 
	}
	else 
	{ 
		this->used_raw_noise.resize(this->orig_num_echos);
	}
	
	this->inv_config.noise_amp = 1.0;
	std::fill_n(t2_amps_ptr, num_total_t2_bins, 0.0);

	double* noise_start_ptr;
	for (int ii = 0; ii < num_inversions; ii++)
	{
		noise_start_ptr = raw_noise_ptr + (ii * this->orig_num_echos);
		std::copy(noise_start_ptr, noise_start_ptr + this->orig_num_echos,
			this->used_raw_noise.data());

		this->invert(raw_amps_size, raw_amps_ptr);
		
		for (int ib = 0; ib < num_total_t2_bins; ib++)
		{
			t2_amps_ptr[ib] += this->used_t2_amps[ib];
		}
	}

	for (int ib = 0; ib < num_total_t2_bins; ib++)
	{
		t2_amps_ptr[ib] /= num_inversions;
	}

	this->inv_config.noise_amp = prev_noise_std;
	if (prev_noise_std) 
	{ 
		this->used_raw_noise.assign(prev_noise_amps.begin(), prev_noise_amps.end()); 
	}
	else
	{
		this->used_raw_noise.clear();
	}
}

