#ifndef CPPHELPER
#define CPPHELPER

#include <iostream>
#include <iomanip>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common.hpp"

// This header should only contain routines that can only be compiled by
// gcc48, for example things utilize std::random, armadillo, and LAPACK

void mynormalcpp(double *, double, double, int, unsigned);
void myuniformcpp(double *, int, unsigned);
void myexponentialcpp(double *, int, unsigned);
void mytest(const int);
void fromchebydomain(double, double, int, double*);
void findprojector(double*, int, int, double*);

void chebyspace(double, double, int, double*);
typedef void (*gridgen_fptr)(double, double, int, double*);

///
/// Markov discretization
///
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

template <class T>
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
void tauchen_vec(int, int, int, double*, double*, double*, double*, gridgen_fptr);

// void qzdecomp(arma::mat &, arma::mat &, arma::mat &, arma::mat &);
// void test();
// void linearQZ(double*, double*, double*, double*, int, int, int, double*);

double levelOLS(double* , double** , int , int , double* );
double   logOLS(double* , double** , int , int , double* );

////////////////////////////////////////
//
/// Utilities for Vector
//
////////////////////////////////////////
void linspace(double min, double max, int N, double* grid);

// A function template to display vectors, C array style
template <class T>
void display_vec(T vec, int size) {
	for (int i = 0; i < size; i++) {
		std::printf("The %ith element, @[%i] = %f\n", i+1, i, (double) vec[i]);
	};
};

// A function template to display vectors, std::vector style
template <class T>
void display_vec(T vec) {
	int size = vec.size();
	for (int i = 0; i < size; i++) {
		std::printf("The %ith element, @[%i] = %f\n", i+1, i, (double) vec[i]);
	};
};

// A function template to save vectors to file, C array style
template <class T>
void save_vec(T vec, int size, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Saving to " << filename << std::endl;
	std::ofstream fileout(filename.c_str(), std::ofstream::trunc);
	for (int i = 0; i < size; i++) {
		fileout << std::setprecision(16) << vec[i] << '\n';
	};
	fileout.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;

};

// A function template to save vectors to file, std::vector style
template <class T>
void save_vec(T vec, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Saving to " << filename << std::endl;
	int size = vec.size();
	std::ofstream fileout(filename.c_str(), std::ofstream::trunc);
	for (int i = 0; i < size; i++) {
		fileout << std::setprecision(16) << vec[i] << '\n';
	};
	fileout.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;
};

// A function template to save vectors to file, C array style
template <class T>
void load_vec(T& vec, int size, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Loading from " << filename << std::endl;
	std::ifstream filein(filename.c_str());
	for (int i = 0; i < size; i++) {
		filein >> vec[i];
	};
	filein.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;
};

// A function template to save vectors to file, vector style
template <class T>
void load_vec(T& vec, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Loading from " << filename << std::endl;
	std::ifstream filein(filename.c_str());
	double temp;
	int N = vec.size();
	for (int i = 0; i < N; i++) {
		filein >> temp;
		vec[i] = temp;
	};
	filein.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;
};
#endif
