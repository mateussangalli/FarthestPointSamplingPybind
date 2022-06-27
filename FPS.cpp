#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

//  struct buffer_info {
//      void *ptr;
//      py::ssize_t itemsize;
//      std::string format;
//      py::ssize_t ndim;
//      std::vector<py::ssize_t> shape;
//      std::vector<py::ssize_t> strides;
//  };

py::array_t<int> farthest_point_sampling(py::array_t<float> distance_matrix, int n_points) {
    py::buffer_info buf = distance_matrix.request();
    int first_point = std::rand() % n_points; // choose a random point as seed

    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");

    if (buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Distance matrix must be square");

    int N = buf.shape[0];
    if (N <= n_points) {
        throw std::runtime_error("Number of sampled points must be smaller than number of input points");
    }

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto indices = py::array_t<int>(n_points);

    py::buffer_info buf_out = indices.request();

    float *ptr_in = static_cast<float *>(buf.ptr);
    int *ptr_out = static_cast<int *>(buf_out.ptr);
    ptr_out[0] = first_point;

    int remaining_points[N];
    int n_rmng = N-1;
    int idx = 0;
    int v = 0;
    while (idx < n_rmng) {
        if (v == first_point) {
            v++;
        }else {
            remaining_points[idx] = v;
            idx++;
            v++;
        }
    }


    float min = 100000.f;
    float max = 0.f;
    int max_ind = 0;
    float current_dist;

    for (size_t idx_out = 1; idx_out < n_points; idx_out++) {
        max = 0.f;
        max_ind = -1;
        for (int idx_rmng = 0; idx_rmng < n_rmng; idx_rmng++) {
            for (size_t idx = 0; idx < idx_out; idx++){
                current_dist = ptr_in[N*ptr_out[idx] + idx_rmng];
                min = 999999.f;
                if (current_dist < min) min = current_dist;
            }
            if (min > max) {
                max = min;
                max_ind = idx_rmng;
            }
        } 
        remaining_points[max_ind] = remaining_points[n_rmng-1];
        n_rmng--;
        ptr_out[idx_out] = max_ind;
    }
    return indices;
}

PYBIND11_MODULE(FPS, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Perform farthest point sampling on the distance matrix");
}
