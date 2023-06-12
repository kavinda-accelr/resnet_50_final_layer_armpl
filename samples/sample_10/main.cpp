#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>

#include <armpl.h>
#include <cblas.h>
#include <omp.h>

void print_arr(float *arr, size_t rows, size_t columns)
{
    for (size_t i = 0; i < rows; i++)
    {
        std::cout << " | ";
        for (size_t j = 0; j < columns; j++)
        {
            std::cout << std::setw(10) << arr[(columns * i) + j];
        }
        std::cout << " | " << std::endl;
    }
    std::cout << std::endl;
}

void test()
{
    // w(number of classes x LLOS) x Trans_i(LLOS x frams) -> o(number of classes x frams)

    // R x C
    // i_vec_size = 3, num frams = 2, num classes = 4
    // i_vec_1 = {13, 14, 15}, i_vec_2 = {16, 17, 18}

    uint32_t i_vec_size = 3;
    uint32_t num_frams = 2;
    uint32_t num_classes = 4;

    float w[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float i[] = {13, 14, 15, 16, 17, 18};
    float b[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};

    float b_test[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};

    print_arr(w, num_classes, i_vec_size);
    print_arr(i, num_frams, i_vec_size);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_classes, num_frams, i_vec_size, 1.0f, w, i_vec_size, i, i_vec_size, 1.0f, b, num_classes);

    cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, i_vec_size, 1.0f, w, i_vec_size, i, 1, 1.0f, b_test, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, i_vec_size, 1.0f, w, i_vec_size, i + i_vec_size, 1, 1.0f, b_test + num_classes, 1);

    print_arr(b, num_frams, num_classes);
    print_arr(b_test, num_frams, num_classes);

    /*
    |   86 104 |
    |  212 257 |
    |  338 410 |
    |  464 563 |

    |         86       212       338       464 |
    |        104       257       410       563 |
    */
}

template <typename T>
std::ostream &operator<<(std::ostream &ostr, const std::vector<T> &vec)
{
    for (const T &item : vec)
    {
        ostr << std::setw(10) << item;
    }
    return ostr;
}

template <typename T>
std::ostream &operator<<(std::ostream &ostr, const std::vector<std::vector<T>> &vecs)
{
    for (const std::vector<T> &vec : vecs)
    {
        ostr << " | ";
        for (const T &item : vec)
        {
            ostr << std::setw(10) << item;
        }
        ostr << " | " << std::endl;
    }

    return ostr;
}

int main()
{
    uint32_t i_vec_size = 3;
    uint32_t num_frams = 2;
    uint32_t num_classes = 4;

    std::vector<float> bias_vec{0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> bias_mat;
    std::vector<std::vector<float>> bias_vecs;

    for (int frame = 0; frame < num_frams; frame++)
    {
        for (const float &bias : bias_vec)
        {
            bias_mat.push_back(bias);
        }
    }

    for (int frame = 0; frame < num_frams; frame++)
    {
        bias_vecs.push_back(bias_vec);
    }

    std::cout << bias_vec << std::endl;
    std::cout << bias_mat << std::endl;
    print_arr(bias_mat.data(), num_frams, num_classes);
    std::cout << bias_vecs << std::endl;

    return 0;
}