#include "TikhonovSolver.h"

TikhonovSolver::TikhonovSolver() 
{
	this->set = false;
}

TikhonovSolver::TikhonovSolver(const MatrixXd &inv_matrix, const vector<double> &B,
	const double min_lambda, const double max_lambda, const int num_lambdas)
{
	this->find_best_lambda(inv_matrix, B, min_lambda, max_lambda, num_lambdas);
}

void TikhonovSolver::find_best_lambda(const MatrixXd &inv_matrix, const vector<double> &B,
	const double min_lambda, const double max_lambda, const int num_lambdas)
{
	//BDCSVD<MatrixXd> svd(inv_matrix, ComputeThinU);
	JacobiSVD<MatrixXd> svd(inv_matrix, ComputeThinU);
	MatrixXd matrixU = svd.matrixU();
	VectorXd s_values = svd.singularValues();
	const vector<double> all_singular_values(s_values.data(), s_values.data() + s_values.size());
	
	double svalue;
	this->singular_values.clear();
	for (int sind = 0; sind < all_singular_values.size(); sind++)
	{
		svalue = all_singular_values[sind];
		if (svalue) { this->singular_values.push_back(svalue); }
		else { break; }
	}

	const int num_singular_values = singular_values.size();
	this->uatb_array.resize(num_singular_values, 0.0);

	const int num_cpu_threads = omp_get_max_threads();

	#pragma omp parallel num_threads(num_cpu_threads)
	{
		int j;
		double amp_value;

		int j_start, j_finish;
		const int tid = omp_get_thread_num();
		get_multi_thread_loop_limits(tid, num_cpu_threads, num_singular_values, j_start, j_finish);

		for (int i = 0; i < B.size(); i++)
		{
			amp_value = B[i];
			for (j = j_start; j < j_finish; j++)
			{
				this->uatb_array[j] += (matrixU(i, j) * amp_value);
			}
		}
	}
	this->curvature.resize(num_lambdas);
	this->solution_norms.resize(num_lambdas);
	this->residual_norms.resize(num_lambdas);
	this->lambdas = logspace_vector(min_lambda, max_lambda, num_lambdas);
	this->uatb_array_divsvs = div_vector(this->uatb_array, singular_values);
	this->square_uatb_array_divsvs = square_vector(uatb_array_divsvs);

	#pragma omp parallel num_threads(num_cpu_threads)
	{
		int lmb_start, lmb_finish;
		const int tid = omp_get_thread_num();
		get_multi_thread_loop_limits(tid, num_cpu_threads, num_lambdas, lmb_start, lmb_finish);

		LcurveValues lc_values;
		for (int il = lmb_start; il < lmb_finish; il++)
		{
			lc_values = calc_Lcurve_values_from_lambda(this->lambdas[il]);
			this->solution_norms[il] = lc_values.solution_norm;
			this->residual_norms[il] = lc_values.residual_norm;
			this->curvature[il] = lc_values.curvature;
		}
	}
	this->best_lambda_index = get_vector_max_element(this->curvature);
	this->best_lambda = this->lambdas[this->best_lambda_index];
	this->set = true;
}

void TikhonovSolver::set_Lcurve_values_from_lambda(const double lambda, 
	double& solution_norm, double& residual_norm, double& curvature)
{
	const double sql = lambda * lambda;
	double eta_sum = 0.0;
	double rho_sum = 0.0;
	double deta_sum = 0.0;

	double mult, numerator, denominator;
	double sq_sgv, ft, onem_ft, ft_uatb_divsvs, onem_ft_uatb;

	for (int isv = 0; isv < singular_values.size(); isv++)
	{
		sq_sgv = this->singular_values[isv];
		sq_sgv *= sq_sgv;

		ft = sq_sgv / (sq_sgv + sql);
		onem_ft = 1.0 - ft;
		ft_uatb_divsvs = (ft * this->uatb_array_divsvs[isv]);
		onem_ft_uatb = onem_ft * this->uatb_array[isv];

		eta_sum += ft_uatb_divsvs * ft_uatb_divsvs;
		rho_sum += onem_ft_uatb * onem_ft_uatb;
		deta_sum += onem_ft * ft * ft * this->square_uatb_array_divsvs[isv];
	}

	deta_sum *= -4.0 / lambda;
	mult = -2.0 * eta_sum * rho_sum / deta_sum;
	numerator = (sql * deta_sum * rho_sum) + (2.0 * lambda * eta_sum * rho_sum) + (sql * sql * eta_sum * deta_sum);
	denominator = pow((sql * eta_sum * eta_sum) + (rho_sum * rho_sum), 1.5);

	solution_norm = sqrt(eta_sum);
	residual_norm = sqrt(rho_sum);
	curvature = mult * numerator / denominator;
}

LcurveValues TikhonovSolver::calc_Lcurve_values_from_lambda(const double lambda)
{
	LcurveValues return_values;
	this->set_Lcurve_values_from_lambda(lambda, return_values.solution_norm,
		return_values.residual_norm, return_values.curvature);
	return return_values;
}