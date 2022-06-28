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
    Center *parent;
};

struct CompareDistance {
    bool operator()(ServedPoint const& a, ServedPoint const& b) {
        return (a.distance < b.distance);
    }
};


struct Center {
    int index;
    std::list<Center*> friends;
    std::list<ServedPoint*> served_points;
};


float sqr_eucl_distance(float *a, float *b, int d) {
    float out = 0.;
    for (int idx = 0; idx < d; idx++) {
        out += (a[idx] - b[idx]) * (a[idx] - b[idx]);
    }
    return out;
}

py::array_t<int> farthest_point_sampling(py::array_t<float> points, int n_points, int first_point) {
    py::buffer_info buf = points.request();

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

    // initialize centers
    std::vector<Center> centers(n_points);
    for (int idx=0; idx < n_points; idx++) {
        //Center *temp = (Center*)malloc(sizeof(Center));
        Center c0;
        std::list<ServedPoint*> temp1;
        std::list<Center*> temp2;
        c0.served_points = temp1;
        c0.friends = temp2;
        centers[idx] = c0;
    }
    centers[0].index = first_point;
    centers[0].friends.push_back(&centers[0]);

    float max_dist = 0.;
    float dist;
    // fursthest served point from the first point
    ServedPoint *p0;
    // initialize vector of served points by the first point
    for (int idx = 0; idx < N; idx++) {
        dist = sqr_eucl_distance(ptr_in + first_point * d, ptr_in + idx * d, d);  
        ServedPoint* point = (ServedPoint*)malloc(sizeof(ServedPoint));
        point->distance = dist;
        point->index = idx;
        point->parent = &centers[0];
        if (dist > max_dist) {
            p0 = point;
            max_dist = dist;
        }
        centers[0].served_points.push_back(point);
    }
    std::priority_queue<ServedPoint, std::vector<ServedPoint>, CompareDistance> furthest_served_points;
    furthest_served_points.push(*p0);


    float r = 999999.;

    std::list<Center*>::iterator it_friends;
    std::list<ServedPoint*>::iterator it_sp;
    float dist_new;
    float dist_centers_old;
    float dist_centers_new;
    ServedPoint p, p1, p2;
    Center *c_old;
    for (int idx = 1; idx < n_points; idx ++) {
        // find next center points
        p = furthest_served_points.top();
        furthest_served_points.pop();
        ptr_out[idx] = p.index;
        r = p.distance;

        c_old = p.parent;


        centers[idx].index = p.index;
        centers[idx].friends.push_back(&centers[idx]);

        // delete friends
        it_friends = c_old->friends.begin();
        while (it_friends != c_old->friends.end()) {
            dist_centers_old = sqr_eucl_distance(ptr_in + (d * c_old->index), ptr_in + (d * ((*it_friends)->index)), d);
            if (dist_centers_old > 64 * r) {
                // it_friends->friends.erase(&c_old);
                c_old->friends.erase(it_friends++);
            }
            else it_friends++;
        }

        // update served points
        it_friends = c_old->friends.begin();
        while (it_friends != c_old->friends.end()) {
            it_sp = (*it_friends)->served_points.begin();
            while(it_sp != (*it_friends)->served_points.end()) {
                dist_new = sqr_eucl_distance(ptr_in + (d * p.index), ptr_in + (d * ((*it_sp)->index)), d);
                if (dist_new < (*it_sp)->distance) {
                    (*it_sp)->distance = dist_new;
                    (*it_sp)->parent = &centers[idx];
                    centers[idx].served_points.push_back(*it_sp);
                    (*it_friends)->served_points.erase(it_sp++);
                }else {
                    it_sp++;
                }
            }
            it_friends++;
        }
        // add friends
        it_friends = c_old->friends.begin();
        while (it_friends != c_old->friends.end()) {
            dist_centers_new = sqr_eucl_distance(ptr_in + (d * p.index), ptr_in + (d * ((*it_friends)->index)), d);
            if (dist_centers_new < 16 * r) {
                centers[idx].friends.push_back(*it_friends);
                (*it_friends)->friends.push_back(&centers[idx]);
            }
            it_friends++;
        }

        max_dist = 0.;
        for (it_sp = c_old->served_points.begin(); it_sp != c_old->served_points.end(); it_sp++) {
            if ((*it_sp) -> distance > max_dist){
                p1 = **it_sp;
                max_dist = (*it_sp) -> distance;
            }
        }
        max_dist = 0.;
        for (it_sp = centers[idx].served_points.begin(); it_sp != centers[idx].served_points.end(); it_sp++) {
            if ((*it_sp) -> distance > max_dist){
                p2 = **it_sp;
                max_dist = (*it_sp) -> distance;
            }
        }
        furthest_served_points.push(p1);
        furthest_served_points.push(p2);
    }
    for (int idx=0; idx < n_points; idx++) {
        // free(centers[idx]);
        for (it_sp=centers[idx].served_points.begin(); it_sp != centers[idx].served_points.end(); it_sp++) {
            free(*it_sp);
        }
    }

    return indices;
}

PYBIND11_MODULE(FPS, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Perform farthest point sampling on the distance matrix");
}
