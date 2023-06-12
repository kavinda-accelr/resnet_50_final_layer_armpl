#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <sstream>
#include <fstream>

#include <armpl.h>
#include <cblas.h>
#include <omp.h>

typedef float tensor_t;

int main()
{
    float weight_mat[] = {1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f};
    float bias_vec[] = {0.1f, 0.2f, 0.3f};
    float x[] = {1.0f, 2.0f};

    // bias_vec := alpha * weight_mat * x + beta * bias_vec
    // (3 x 2) x (2 x 1) = (3 x 1)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 2, 1.0f, weight_mat, 2, x, 1, 1.0f, bias_vec, 1);

    // Print the result
    std::cout << "y = [" << bias_vec[0] << ", " << bias_vec[1] << ", " << bias_vec[2] << "]" << std::endl;

    return 0;
}