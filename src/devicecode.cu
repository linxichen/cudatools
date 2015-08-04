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

// This function fit a valuex x to a grid X of size n. For std::vector like stuff
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).


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


