/*

Author: Pedro Vianna Mesquita

Tikhonov Solver 

Finds the best value for Tikhonov's regularization from a "Ax=B" problem, using an approach described in:
	
	"Hansen, P.C., 1999. The L-curve and its use in the numerical treatment of inverse
	problems. IMM, Department of Mathematical Modelling, Technical University of Denmark."

The class' construtor receives "A" and "B", and the range (min_lambda and max_lambda, at base10) and number (num_lambdas) of regularization values.
"A" must be a Eigen::MatrixXd, while "B" is a std::vector.

When constructed, it creates a logarithmic space of lambdas using these values, and calculates the solution's norm, residual's norm and curvature of the L curve for each lambda.
The results are kept in the member vectors: lambdas, solution_norms, residual_norms and curvature. With them you can plot the L curve and its curvature.
The best regularization (lambda) is usually the one with the biggest curvature value.
You can alse calculate these L curves' values for any other lambda using "set_Lcurve_values_from_lambda" or "calc_Lcurve_values_from_lambda".

It uses Eigen for performing SVD of the "A" matrix (Eigen::JacobiSVD).

*/

#ifndef TIKHONOVSOLVER_H
#define TIKHONOVSOLVER_H

#include <omp.h>
#include <iostream>
#include "vector_funcs.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

//#include <Eigen/src/SVD/BDCSVD.h>
//using Eigen::BDCSVD;

using std::cout;
using std::endl;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::JacobiSVD;
using Eigen::ComputeThinU;

#define DEFAULT_MIN_LAMBDA -4.0  //0.0001
#define DEFAULT_MAX_LAMBDA 2.0   //100.0
#define DEFAULT_NUM_LAMBDAS 512

struct LcurveValues
{
	double solution_norm, residual_norm, curvature;
};


class TikhonovSolver
{
public:
	bool set;
	double best_lambda;
	int best_lambda_index;
	vector<double> singular_values;
	vector<double> uatb_array, uatb_array_divsvs, square_uatb_array_divsvs;
	vector<double> curvature, lambdas, solution_norms, residual_norms;

	TikhonovSolver();

	TikhonovSolver(
		const MatrixXd &inv_matrix, const vector<double> &B,
		const double min_lambda = DEFAULT_MIN_LAMBDA, 
		const double max_lambda = DEFAULT_MAX_LAMBDA, 
		const int num_lambdas = DEFAULT_NUM_LAMBDAS
	);

	void find_best_lambda(
		const MatrixXd &inv_matrix, const vector<double> &B,
		const double min_lambda = DEFAULT_MIN_LAMBDA, 
		const double max_lambda = DEFAULT_MAX_LAMBDA, 
		const int num_lambdas = DEFAULT_NUM_LAMBDAS
	);

	void set_Lcurve_values_from_lambda(const double lambda, 
		double& solution_norm, double& residual_norm, double& curvature);

	LcurveValues calc_Lcurve_values_from_lambda(const double lambda);
};

#endif // !TIKHONOVSOLVER_H
