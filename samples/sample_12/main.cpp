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

void setup_bias_mat(const std::vector<float>& bias_vec, std::vector<float>& bias_mat, uint32_t num_frams)
{
    for (uint32_t frame = 0; frame < num_frams; frame++)
    {
        for (const float &bias : bias_vec)
        {
            bias_mat.push_back(bias);
        }
    }
}

void setup_bias_vecs(const std::vector<float>& bias_vec, std::vector<std::vector<float>>& bias_vecs, uint32_t num_frams)
{
    for (uint32_t frame = 0; frame < num_frams; frame++)
    {
        bias_vecs.push_back(bias_vec);
    }
}

void setup_input_mat(const std::vector<std::vector<float>>& input_vecs, std::vector<float>& input_mat)
{
    for (const std::vector<float> &input_vec : input_vecs)
    {
        input_mat.insert(input_mat.end(), input_vec.begin(), input_vec.end());
    }
}

int main()
{
    uint32_t input_vec_size = 4;
    uint32_t num_frams = 3;
    uint32_t num_classes = 5;

    const std::vector<float> bias_vec{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    const std::vector<std::vector<float>> input_vecs{{21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}};
    const std::vector<float> weight_mat{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    std::vector<float> bias_mat;
    std::vector<std::vector<float>> bias_vecs;
    std::vector<float> input_mat;

    setup_bias_mat(bias_vec, bias_mat, num_frams);
    setup_bias_vecs(bias_vec, bias_vecs, num_frams);
    setup_input_mat(input_vecs, input_mat);

    for (uint32_t frame = 0; frame < num_frams; frame++)
    {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, input_vec_size, 1.0f, weight_mat.data(), input_vec_size, input_vecs[frame].data(), 1, 1.0f, bias_vecs[frame].data(), 1);
    }

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_classes, num_frams, input_vec_size, 1.0f, weight_mat.data(), input_vec_size, input_mat.data(), input_vec_size, 1.0f, bias_mat.data(), num_classes);
    
    std::cout << bias_vecs << std::endl;
    print_arr(bias_mat.data(), num_frams, num_classes);

    return 0;
}