#ifndef COMMONHEADER
#define COMMONHEADER

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

/// Header file that contains things many files needs to know

/// typedef useful things
typedef thrust::host_vector<double>   h_vec_d;
typedef thrust::host_vector<int>      h_vec_i;
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

	/// default constructor
	cudavec() {
		n = 1;
		hvec = thrust::host_vector<T>(n,0.0);
		dvec = thrust::device_vector<T>(n,0.0);
		hptr = thrust::raw_pointer_cast(hvec.data());
		dptr = thrust::raw_pointer_cast(dvec.data());
	};

	/// constructor, with random default value of size
	cudavec(size_t _n, T x) {
		n = _n;
		hvec = thrust::host_vector<T>(n,x);
		dvec = thrust::device_vector<T>(n,x);
		hptr = thrust::raw_pointer_cast(hvec.data());
		dptr = thrust::raw_pointer_cast(dvec.data());
	}

	/// constructor, with specified value x
	cudavec(size_t _n) {
		n = _n;
		hvec = thrust::host_vector<T>(n);
		dvec = thrust::device_vector<T>(n);
		hptr = thrust::raw_pointer_cast(hvec.data());
		dptr = thrust::raw_pointer_cast(dvec.data());
	}

	// get size
	size_t size() {
		return n;
	}

	/// copy from host to device
	void h2d() {
		dvec = hvec;
	};

	/// copy from device to host
	void d2h() {
		hvec = dvec;
	};

	///
	/// on the host, wrap the [] operator, you can directly write
	/// the host_vector. pretty bad practice though. */
	///
	T& operator[] (const unsigned int i) {
		return hptr[i];
	}

	/// assignment operator

};
#endif
