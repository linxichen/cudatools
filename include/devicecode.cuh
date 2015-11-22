#ifndef __CUDAHELPER__
#define __CUDAHELPER__

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common.hpp"

////////////////////////////////////////
//
/// Interpolation Stuff
//
////////////////////////////////////////
// Linear interpolation
__host__ __device__
double linear_interp(double , double , double , double , double );

// Bilinear interpolation
__host__ __device__
double bilinear_interp(double , double , double , double , double , double , double , double , double , double );

// This function converts index to subscripts like ind2sub in MATLAB
// Purpose:		Converts index to subscripts. i -> [i_1, i_2, ..., i_n]
//
// Input:		length_size = # of coordinates, i.e. how many subscripts you are getting
// 				siz_vec = vector that stores the largest coordinate value for each subscripts. Or the dimensions of matrices
// 				index = the scalar index
//
// Ouput:		subs = the vector stores subscripts
__host__ __device__
void ind2sub(int length_size, int* siz_vec, int index, int* subs);

// This function fit a valuex x to a increasing grid X of size n.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
__host__ __device__
int fit2grid(const double x, const int n, const double* X);

// This function fit a valuex x to a increasing grid X of size n.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
// grid is accessed with stride s. we are looking at j = 1:n X[stride+j*n]
__host__ __device__
int fit2grid(const double x, const int n, const double* X, const int stride);

// This function fit a valuex x to a "even" grid X of size n. Even means equi-distance among grid points.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
__host__ __device__
int fit2evengrid(const double x, const int n, const double min, const double max);

/// This function fit a valuex x to a grid X of size n.
/// For std::vector like stuff
/// The largest value on grid X that is smaller than x is
/// returned ("left grid point" is returned).
template <class T>
int fit2grid(const double x, const T X);

////////////////////////////////////////
//
// Nonlinear Equation Solver
//
////////////////////////////////////////
// Newton's Method with bracketing, i.e. we know on two points the function differs in sign.
// Codes from Numerical Recipes 3rd. Ed.
// BEWARE: The stopping criteria is not right yet.
template <class T>
__host__ __device__
double newton_bracket(T func, double x1, double x2, double x0);

/// Purpose: Tries to find a root for function named func.
/// Its first derivative is given by func.prime().
/// func is only defined on [x1,x2] We "pull back" when outside.
/// x0 is the guess.
template <class T>
__host__ __device__
double newton(T func, const double x1, const double x2, double x0);

////////////////////////////////////////
//
// Chebyshev Toolset
//
////////////////////////////////////////
// Evaluate Chebychev polynomial of any degree
__host__ __device__
double chebypoly(const int p, const double x);

// Evaluate Chebychev polynomial of any degree
__host__ __device__
int chebyroots(const int p, double* roots);

// Evaluate Chebychev approximation of any degree
	// Note that coefficient vector has p+1 values
__host__ __device__
double chebyeval(int p, double x, double* coeff) ;

// Eval multi-dimensional Chebyshev tensor basis
// y = sum T_pi(x_i), i = 1,2,...p
__host__ __device__
double chebyeval_multi (const int n_var, double* x, int* size_vec,int* temp_subs, double* coeff) ;

////////////////////////////////////////
//
// Some tools for simulation
//
////////////////////////////////////////

// a quick and dirty exclusive scan to turn
// markov transition matrix into CDF matrix
__host__ __device__
void pdf2cdf(double* P, size_t n, double* CDF);

// draw from a n-state discrete distribution, wit aug-CDF
// given as a n-by-n array begins with 0 and then cumsum(PDF(1:end)).
// e.g at i_now = 0, PDF(i_now,:) = [0.2 0.3 0.5], then CDF[i_now,:] = [0 0.2 0.5]
// also given a random number in [0,1] for inverse CDF.
/// highly recomment to use with pdf2cdf function.
__host__ __device__
int markovdiscrete(int i_now, double* CDF, size_t n, double u);

__host__ __device__
void markovsimul(int T, double* CDF, int n, double* u, int init, int* sim);

#endif
