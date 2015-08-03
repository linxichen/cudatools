#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "../include/common.hpp"

template<typename T>
cudavec<T>::cudavec(size_t _n) {
	n = _n;
	hvec = thrust::host_vector<T>(n);
	dvec = thrust::device_vector<T>(n);
	hptr = thrust::raw_pointer_cast(hvec.data());
	dptr = thrust::raw_pointer_cast(dvec.data());
}

template<typename T>
cudavec<T>::cudavec(size_t _n, T x) {
	n = _n;
	hvec = thrust::host_vector<T>(n,x);
	dvec = thrust::device_vector<T>(n,x);
	hptr = thrust::raw_pointer_cast(hvec.data());
	dptr = thrust::raw_pointer_cast(dvec.data());
}

template<typename T>
size_t cudavec<T>::size() {
	return n;
}

template<typename T>
void cudavec<T>::h2d() {
	dvec = hvec;
}

template<typename T>
void cudavec<T>::d2h() {
	hvec = dvec;
}

template<typename T>
T& cudavec<T>::operator[] (const unsigned int i){
	return hptr[i];
}
