#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "../include/common.hpp"

template<typename T>
cudavec::cudavec(size_t _n, T x) {
		n = _n;
		hvec = thrust::host_vector<T>(n);
		dvec = thrust::device_vector<T>(n);
		hptr = thrust::raw_pointer_cast(hvec.data());
		dptr = thrust::raw_pointer_cast(dvec.data());
}
