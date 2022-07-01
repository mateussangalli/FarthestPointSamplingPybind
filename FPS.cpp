#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <list>
#include <queue>

namespace py = pybind11;

struct Center;

struct ServedPoint {
    float distance;
    int index;
    int parent;
};

struct CompareDistance {
    bool operator()(ServedPoint const& a, ServedPoint const& b) {
        return (a.distance < b.distance);
    }
};

ServedPoint find_max_served_point(std::vector<ServedPoint> points, std::list<int> indices) {
    float max_dist = -1.;
    std::list<int>::iterator it_sp;
    ServedPoint p;
    for (it_sp = indices.begin(); it_sp != indices.end(); it_sp++) {
        if (points[*it_sp].distance > max_dist){
            p = points[*it_sp];
            max_dist = points[*it_sp].distance;
        }
    }
    return p;

}


float sqr_eucl_distance(float *a, float *b, int d) {
    float out = 0.;
    for (int idx = 0; idx < d; idx++) {
        out += (a[idx] - b[idx]) * (a[idx] - b[idx]);
    }
    return out;
}

py::array_t<int> farthest_point_sampling(py::array_t<float> input_points, int n_points, int first_point) {
    py::buffer_info buf = input_points.request();

    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");


    int N = buf.shape[0];
    int d = buf.shape[1];
    if (N <= n_points) {
        throw std::runtime_error("Number of sampled points must be smaller than number of input points");
    }

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto indices = py::array_t<int>(n_points);

    py::buffer_info buf_out = indices.request();

    float *ptr_in = static_cast<float *>(buf.ptr);
    int *ptr_out = static_cast<int *>(buf_out.ptr);
    ptr_out[0] = first_point;

    // indexes of center points
    std::vector<int> centers(n_points);
    centers[0] = first_point;

    std::vector<ServedPoint> all_points(N);
    std::vector<std::list<int>> friends(n_points, std::list<int>(0));
    std::vector<std::list<int>> served_points(n_points, std::list<int>(0));

    friends[0].push_back(0);

    float max_dist = 0.;
    int max_ind = -1;
    float dist;
    // initialize vector of served points by the first point
    for (int idx = 0; idx < N; idx++) {
        dist = sqr_eucl_distance(ptr_in + first_point * d, ptr_in + idx * d, d);  
        ServedPoint point = {dist, idx, 0};
        all_points[idx] = point;
        served_points[0].push_back(idx);
        if (dist > max_dist) {
            max_dist = dist;
            max_ind = idx;
        }
    }
    std::priority_queue<ServedPoint, std::vector<ServedPoint>, CompareDistance> furthest_served_points;
    ServedPoint p0 = {max_dist, max_ind, 0};
    furthest_served_points.push(p0);


    float r = 999999.;

    std::list<int>::iterator it_friends;
    std::list<int>::iterator it_sp;
    float dist_new;
    float dist_centers_old;
    float dist_centers_new;
    ServedPoint p;
    int c_old;
    for (int idx = 1; idx < n_points; idx ++) {
        p = furthest_served_points.top();

        while (p.parent != all_points[p.index].parent) {
            furthest_served_points.pop();
            furthest_served_points.push(find_max_served_point(all_points, served_points[p.parent]));
            p = furthest_served_points.top();
        }
        furthest_served_points.pop();
        ptr_out[idx] = p.index;
        r = p.distance;

        c_old = p.parent;

        centers[idx] = p.index;
        friends[idx].push_back(idx);

        // delete friends
        it_friends = friends[c_old].begin();
        while (it_friends != friends[c_old].end()) {
            dist_centers_old = sqr_eucl_distance(ptr_in + (d * centers[c_old]), ptr_in + (d * centers[*it_friends]), d);
            if (dist_centers_old > 64 * r) {
                friends[c_old].erase(it_friends++);
            }
            else it_friends++;
        }

        // update served points
        it_friends = friends[c_old].begin();
        while (it_friends != friends[c_old].end()) {
            it_sp = served_points[*it_friends].begin();
            while(it_sp != served_points[*it_friends].end()) {
                dist_new = sqr_eucl_distance(ptr_in + (d * p.index), ptr_in + (d * (all_points[*it_sp].index)), d);
                if (dist_new <= all_points[*it_sp].distance) {
                    all_points[*it_sp].distance = dist_new;
                    all_points[*it_sp].parent = idx;
                    served_points[idx].push_back(*it_sp);
                    served_points[*it_friends].erase(it_sp++);
                }else {
                    it_sp++;
                }
            }
            it_friends++;
        }

        // add friends
        it_friends = friends[c_old].begin();
        while (it_friends != friends[c_old].end()) {
            dist_centers_new = sqr_eucl_distance(ptr_in + (d * p.index), ptr_in + (d * (centers[*it_friends])), d);
            if (dist_centers_new < 16 * r) {
                friends[idx].push_back(*it_friends);
                friends[*it_friends].push_back(idx);
            }
            it_friends++;
        }

        furthest_served_points.push(find_max_served_point(all_points, served_points[c_old]));
        furthest_served_points.push(find_max_served_point(all_points, served_points[idx]));
    }
    while (!furthest_served_points.empty()) {
        furthest_served_points.pop();
    }
    all_points.clear();
    centers.clear();
    return indices;

}

PYBIND11_MODULE(FPS, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Perform farthest point sampling on the distance matrix");
}
