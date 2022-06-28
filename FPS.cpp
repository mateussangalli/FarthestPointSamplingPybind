#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;


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
    float remaining_dists[N];

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

    for (int idx = 0; idx < N; idx++) {
        remaining_dists[idx] = 9999999.;
    }


    float max = 0.f;
    int max_ind = 0;
    int last_ind = first_point;

    for (size_t idx_out = 1; idx_out < n_points; idx_out++) {
        max = 0.f;
        max_ind = -1;
        for (int idx_rmng = 0; idx_rmng < n_rmng; idx_rmng++) {
            // update distances array
            if (ptr_in[N * remaining_points[idx_rmng] + last_ind] < remaining_dists[idx_rmng]) {
                remaining_dists[idx_rmng] = ptr_in[N * idx_rmng + last_ind];
            }
            if (remaining_dists[idx_rmng] > max) {
                max = remaining_dists[idx_rmng];
                max_ind = remaining_points[idx_rmng];
            }
        } 
        n_rmng--;
        remaining_points[max_ind] = remaining_points[n_rmng];
        remaining_dists[max_ind] = remaining_dists[n_rmng];
        last_ind = max_ind;
        ptr_out[idx_out] = max_ind;
    }
    return indices;
}

PYBIND11_MODULE(FPS, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Perform farthest point sampling on the distance matrix");
}
