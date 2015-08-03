#ifndef COMMONHEADER
#define COMMONHEADER

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/// Header file that contains things many files needs to know

/// typedef useful things
typedef thrust::host_vector<double> h_vec_d;
typedef thrust::host_vector<int>    h_vec_i;
typedef thrust::device_vector<double> d_vec_d;
typedef thrust::device_vector<int>    d_vec_i;

// A struct that encompasses host and device vectors
template<class T>
struct cudavec {
	/// data members contains host and device vectors
	size_t n;
	thrust::host_vector<T> hvec;
	thrust::device_vector<T> dvec;
	T* hptr;
	T* dptr;

	/// constructor, with random default value of size
	cudavec(size_t);

	/// constructor, with specified value x
	cudavec(size_t _n, T x);

	// get size
	size_t size();

	/// copy from host to device
	void h2d();

	/// copy from device to host
	void d2h();

	///
	/// on the host, wrap the [] operator, you can directly write
	/// the host_vector. pretty bad practice though. */
	///
	T& operator[] (const unsigned int );
};
#endif
