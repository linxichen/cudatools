#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include "../include/common.hpp"
#include "../include/devicecode.cuh"

////////////////////////////////////////
//
/// Interpolation Stuff
//
////////////////////////////////////////
// Linear interpolation
__host__ __device__
double linear_interp(double x, double x_left, double x_right, double f_left, double f_right) {
	if (abs(x_left-x_right)<1e-10) {
		return f_left;
	} else if (x_left > x_right) {
		return -1200981025976;
	} else {
		return f_left + (f_right-f_left)/(x_right-x_left)*(x-x_left);
	};
};

// Bilinear interpolation
__host__ __device__
double bilinear_interp(double x, double y, double x_left, double x_right, double y_low, double y_high, double f_leftlow, double f_lefthigh, double f_rightlow, double f_righthigh) {
	double f_low = linear_interp(x,x_left,x_right,f_leftlow,f_rightlow);
	double f_high = linear_interp(x,x_left,x_right,f_lefthigh,f_righthigh);
	return linear_interp(y,y_low,y_high,f_low,f_high);
};

// This function converts index to subscripts like ind2sub in MATLAB
__host__ __device__
void ind2sub(int length_size, int* siz_vec, int index, int* subs) {
// Purpose:		Converts index to subscripts. i -> [i_1, i_2, ..., i_n]
//
// Input:		length_size = # of coordinates, i.e. how many subscripts you are getting
// 				siz_vec = vector that stores the largest coordinate value for each subscripts. Or the dimensions of matrices
// 				index = the scalar index
//
// Ouput:		subs = the vector stores subscripts
	int done = 0;
	for (int i=length_size-1; i>=0; i--) {
		// Computer the cumulative dimension
		int cumdim = 1;
		for (int j=0; j<=i-1; j++) {
			cumdim *= siz_vec[j];
		};
		int temp_sub = (index - done)/cumdim;
		subs[i] = temp_sub;
		done += temp_sub*cumdim;
	};
};

// This function fit a valuex x to a increasing grid X of size n.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
__host__ __device__
int fit2grid(const double x, const int n, const double* X) {
	if (x < X[0]) {
		return 0;
	} else if (x >= X[n-1]) {
		return n-1;
	} else {
		int left=0; int right=n-1; int mid=(n-1)/2;
		while(right-left>1) {
			mid = (left + right)/2;
			if (X[mid]==x) {
				return mid;
			} else if (X[mid]<x) {
				left = mid;
			} else {
				right = mid;
			};
		};
		return left;
	}
};

// This function fit a valuex x to a increasing grid X of size n.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
// grid is accessed with stride s. we are looking at j = 1:n X[stride+j*n]
__host__ __device__
int fit2grid(const double x, const int n, const double* X, const int stride) {
	if (x < X[stride+0*n]) {
		return 0;
	} else if (x >= X[stride+(n-1)*n]) {
		return n-1;
	} else {
		int left=0; int right=n-1; int mid=(n-1)/2;
		while(right-left>1) {
			mid = (left + right)/2;
			if (X[stride+mid*n]==x) {
				return mid;
			} else if (X[stride+mid*n]<x) {
				left = mid;
			} else {
				right = mid;
			};
		};
		return left;
	}
}

// This function fit a valuex x to a "even" grid X of size n. Even means equi-distance among grid points.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
__host__ __device__
int fit2evengrid(const double x, const int n, const double min, const double max) {
	if (x <= min) return 0;
	if (x >= max) return n-1;
	double step = (max-min)/(n-1);
	return floor((x-min)/step);
};


/// This function fit a valuex x to a grid X of size n.
/// For std::vector like stuff
/// The largest value on grid X that is smaller than x is
/// returned ("left grid point" is returned).
template <class T>
int fit2grid(const double x, const T X) {
	int n = X.size();
	return fit2grid( x, n, X);
};

////////////////////////////////////////
//
// Chebyshev Toolset
//
////////////////////////////////////////
// Evaluate Chebychev polynomial of any degree
__host__ __device__
double chebypoly(const int p, const double x) {
	switch (p) {
		case 0: // 0-th order Chebyshev Polynomial
			return 1;
		case 1:
			return x;
		case 2:
			return 2*x*x - 1;
		case 3:
			return 4*x*x*x - 3*x;
	}

	// When p>=4, apply the recurrence relation
	double lag1 = 4*x*x*x -3*x;
	double lag2 = 2*x*x - 1;
	double lag0;
	int distance = p - 3;
	while (distance >= 1) {
		lag0 = 2*x*lag1 - lag2;
		lag2 = lag1;
		lag1 = lag0;
		distance--;
	};
	return lag0;
};

// Evaluate Chebychev polynomial of any degree
__host__ __device__
int chebyroots(const int p, double* roots) {
	for (int i=0; i<p; i++) {
		double stuff = p - 0.5 - 1*i;
		roots[i] = cos(M_PI*(stuff)/(p));
	};

	// Account for the fact that cos(pi/2) is not exactly zeros
	if (p%2) {
		roots[(p-1)/2] = 0;
	};
	return 0;
};

// Evaluate Chebychev approximation of any degree
__host__ __device__
double chebyeval(int p, double x, double* coeff) {
	// Note that coefficient vector has p+1 values
	double sum = 0;
	for (int i=0; i<=p; i++) {
		sum += coeff[i]*chebypoly(i,x);
	};
	return sum;
};

// Eval multi-dimensional Chebyshev tensor basis
// y = sum T_pi(x_i), i = 1,2,...p
__host__ __device__
double chebyeval_multi (const int n_var, double* x, int* size_vec,int* temp_subs, double* coeff) {
	// Note size_vec's elements are p+1 for each var
	int tot_deg = 1;
	for (int i = 0; i < n_var; i++) {
		tot_deg *= (size_vec[i]); // Note there's p+1 coeffs
	};

	double eval = 0;
	for (int index = 0; index < tot_deg; index++) {
		// Perform ind2sub to get current degrees for each var
		ind2sub(n_var, size_vec, index, temp_subs);

		// Find the values at current degrees
		double temp = 1;
		for (int i = 0; i < n_var; i++) {
			// printf("%i th subscript is %i\n",i,temp_subs[i]);
			temp *= chebypoly(temp_subs[i],x[i]);
		};

		// Add to the eval
		eval += (coeff[index]*temp);
	};
	return eval;
};

////////////////////////////////////////////////////////////////////////
///
/// Nonlinear Solver stuff
///
////////////////////////////////////////////////////////////////////////
template <class T>
__host__ __device__
double newton_bracket(T func, double x1, double x2, double x0) {
// Purpose: Tries to find a root for function named func.
// Its first derivative is given by func.prime().
// It is assumed that func(x1) and func(x2) are different in sign so
// a root exists within. x0 is the guess.

	const int newton_maxiter = 100;
	const double newton_tol = 1e-3;
	// Checking the bounds: they need to make sense.
	// Or sometimes the bounds are solutions.
	double f1 = func(x1);
	double f2 = func(x2);
	if (f1*f2>0) return -5179394.1; // The different sign assumption violated!
	if (f1 == 0) return x1;
	if (f2 == 0) return x2;

	// Orient the search so that f(xl) < 0
	double xl, xh;
	if (f1 < 0.0) {
		xl = x1;
		xh = x2;
	} else {
		xh = x1;
		xl = x2;
	};

	// Initialize guess and other things
	double rts = x0;
	double dxold = abs(x2-x1);
	double dx = dxold;
	double f = func(rts);
	double df = func.prime(rts);

	for (int iter = 0; iter < newton_maxiter; iter++) {
		if (
			( ((rts-xh)*df-f)*((rts-xl)*df-f) > 0.0 )   ||	// Bisect if Newton step out of range
			( abs(2.0*f) > abs(dxold*df)  ) // ... or step not decreasing fast enough
		)
		{
			dxold = dx;
			dx = 0.5*(xh-xl);
			rts += dxold; // undo the newton step
			rts = xl + dx;
			if (xl == rts) return rts;
		} else {
			// If newton step is okay
			dxold = dx;
			dx = f/df;
			double temp = rts;
			rts -= dx;
			if (temp==rts) return rts;
		};

		// Check for convergence
		if ( abs(dx)/(1+abs(rts+dx)) < newton_tol ) return rts;

		// Compute new f and df for next iteration
		f = func(rts);
		df = func.prime(rts);

		// Maintain the bracket
		if (f < 0.0) {
			xl = rts;
		} else {
			xh = rts;
		};
	};

	return -51709394.2;
};

template <class T>
__host__ __device__
double newton(T func, const double x1, const double x2, double x0) {
	const int newton_maxiter = 20;
	const double newton_tol = 1e-4;
	// Initialize guess and other things
	double x_old = x0;
	double x = x0;
	double f1 = func(x1);
	double f2 = func(x2);
	if (f1==0) return x1;
	if (f2==0) return x2;
	for (int iter = 0; iter < newton_maxiter; iter++) {
		x = x_old - func(x)/func.prime(x);

		// Pull back if outside of support
		if (x<=x1) {
			return -51709394.2;
		};
		if (x>=x2) {
			return -51709394.2;
		};

		// Check for convergence
		if ( (abs(x-x_old)/(1+abs(x_old))<newton_tol) && (abs(func(x)) < newton_tol) ) {
			return x;
		} else {
			x_old = x;
		};
	};
	return -51709394.2;
};

////////////////////////////////////////
//
// Some tools for simulation
//
////////////////////////////////////////

// a quick and dirty exclusive scan to turn
// markov transition matrix into CDF matrix
__host__ __device__
void pdf2cdf(double* P, size_t n, double* CDF) {
	for (int i_now = 0; i_now < n; i_now++) {
		CDF[i_now+0*n] = 0;
		for (unsigned int i_tmr = 1; i_tmr < n; i_tmr++) {
			CDF[i_now+i_tmr*n] = P[i_now+(i_tmr-1)*n] + CDF[i_now+(i_tmr-1)*n];
		};
	};
};

// draw from a n-state discrete distribution, wit aug-CDF
// given as a n-by-n array begins with 0 and then cumsum(PDF(1:end)).
// e.g at i_now = 0, PDF(i_now,:) = [0.2 0.3 0.5], then CDF[i_now,:] = [0 0.2 0.5]
// also given a random number in [0,1] for inverse CDF.
/// highly recomment to use with pdf2cdf function.
__host__ __device__
int markovdiscrete(int i_now, double* CDF, size_t n, double u) {
	for (unsigned int i_tmr = 1; i_tmr < n; i_tmr++) {
		if ( CDF[i_now+i_tmr*n] > u ) {
			return i_tmr-1;
		}
	};
	return n-1;
};

__host__ __device__
void markovsimul(int T, double* CDF, int n, double* u, int init, int* sim) {
	sim[0] = init;
	for (int t = 1; t < T; t++) {
		sim[t] = markovdiscrete(sim[t-1],CDF,n,u[t]);
	};
};
