#ifndef __CUDAHELPER__
#define __CUDAHELPER__

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

// Tauchen's method for c++ vector style input. only available on host!
// Note must use & so you can actually modify its content
template <class T>
void tauchen(double rrho, double ssigma, T& Z, T& P, double width) {
	// Form the grid and neccessary info
	double ssigma_z = sqrt( pow(ssigma,2)/(1-pow(rrho,2)) );
	int nzgrid = Z.size();
	Z[nzgrid-1] = width*ssigma_z; Z[0] = -width*ssigma_z;
	double step = (Z[nzgrid-1] - Z[0])/ double(nzgrid-1);
	for (int i = 2; i <= nzgrid-1; i++) {
		Z[i-1] = Z[i-2] + step;
	};

	// Find P(i_z,1) and P(i_z,end)
	for (int i_z = 0; i_z <= nzgrid-1; ++i_z) {
		P[i_z + nzgrid*0]          = normcdf( (Z[0]-rrho*Z[i_z]+step/2)/ssigma  );
		P[i_z + nzgrid*(nzgrid-1)] = 1 - normcdf( (Z[nzgrid-1]-rrho*Z[i_z]-step/2)/ssigma  );
	};

	for (int i_z = 0; i_z <= nzgrid-1; ++i_z) {
		for (int i_zplus = 1; i_zplus <= nzgrid-2; ++i_zplus) {
            P[i_z+nzgrid*i_zplus] = normcdf( (Z[i_zplus]-rrho*Z[i_z]+step/2)/ssigma  )-normcdf( (Z[i_zplus]-rrho*Z[i_z]-step/2)/ssigma  );
		};
	};
};

template<class T>
void tauchen_givengrid(double rrho, double ssigma, T& Z, T& P, double width) {
	int nzgrid = Z.size();
	double step = (Z[nzgrid-1] - Z[0])/ double(nzgrid-1);

	for (int i_z = 0; i_z <= nzgrid-1; ++i_z) {
		P[i_z] = normcdf( (Z[0]-rrho*Z[i_z]+step/2)/ssigma  );
		P[i_z + nzgrid*(nzgrid-1)] = 1 - normcdf( (Z[nzgrid-1]-rrho*Z[i_z]-step/2)/ssigma  );
	};

	for (int i_z = 0; i_z <= nzgrid-1; ++i_z) {
		for (int i_zplus = 1; i_zplus <= nzgrid-2; ++i_zplus) {
            P[i_z+nzgrid*i_zplus] = normcdf( (Z[i_zplus]-rrho*Z[i_z]+step/2)/ssigma  )-normcdf( (Z[i_zplus]-rrho*Z[i_z]-step/2)/ssigma  );
		};
	};
};

////////////////////////////////////////
//
/// Interpolation Stuff
//
////////////////////////////////////////
// Linear interpolation
template<typename T>
__host__ __device__ T linear_interp(T , T , T , T , T );

// Bilinear interpolation
template<typename T>
__host__ __device__
T bilinear_interp(T , T , T , T , T , T , T , T , T , T );

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

// This function fit a valuex x to a grid X of size n. For std::vector like stuff
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
template <class T>
int fit2grid(const double x, const T X);

////////////////////////////////////////
//
/// Utilities for Vector
//
////////////////////////////////////////
__host__ __device__
void linspace(double min, double max, int N, double* grid);

// A function template to display vectors, C array style
template <class T>
void display_vec(T vec, int size);

// A function template to display vectors, std::vector style
template <class T>
void display_vec(T vec);

// A function template to save vectors to file, C array style
template <class T>
void save_vec(T vec, int size, std::string filename );

// A function template to save vectors to file, std::vector style
template <class T>
void save_vec(T vec, std::string filename );

// A function template to save vectors to file, C array style
template <class T>
void load_vec(T& vec, int size, std::string filename );

// A function template to save vectors to file, vector style
template <class T>
void load_vec(T& vec, std::string filename );

////////////////////////////////////////
//
// Nonlinear Equation Solver
//
////////////////////////////////////////
// Newton's Method with bracketing, i.e. we know on two points the function differs in sign.
// Codes from Numerical Recipes 3rd. Ed.
// BEWARE: The stopping criteria is not right yet.
// Purpose: Tries to find a root for function named func. Its first derivative is given by func.prime().
//			It is assumed that func(x1) and func(x2) are different in sign so a root exists within. x0 is the guess.
template <class T>
__host__ __device__
double newton_bracket(T func, const double x1, const double x2, double x0);

// "Raw" Newton's Method
// Codes from Numerical Recipes 3rd. Ed.
// Purpose: Tries to find a root for function named func. Its first derivative is given by func.prime().
//			func is only defined on [x1,x2] We "pull back" when outside. x0 is the guess.
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
template<typename T>
__host__ __device__
void pdf2cdf(T* P, size_t n, T* CDF) ;

// draw from a n-state discrete distribution, wit aug-CDF
// given as a n-by-n array begins with 0 and then cumsum(PDF(1:end)).
// e.g at i_now = 0, PDF(i_now,:) = [0.2 0.3 0.5], then CDF[i_now,:] = [0 0.2 0.5]
// also given a random number in [0,1] for inverse CDF.
template<typename T>
__host__ __device__
unsigned int markovdiscrete(unsigned int i_now, T* CDF, size_t n, T u) ;

#endif
