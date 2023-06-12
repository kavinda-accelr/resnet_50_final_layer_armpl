#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>
#include <sstream>
#include <fstream>

#include <armpl.h>
#include <cblas.h>
#include <omp.h>

#include "Timer.h"

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

void vector_populator(const std::string &name, std::vector<float> &vec)
{
    std::ifstream file(name);
    if (!file.is_open())
    {
        std::cout << "File not opened..\n";
        exit(EXIT_FAILURE);
    }

    std::string str;
    while (getline(file, str))
    {
        vec.push_back(std::stof(str));
    }
}

void vec_2_dup_vecs(const std::vector<float> &vec, std::vector<std::vector<float>> &vecs, uint32_t num_frams)
{
    for (uint32_t frame = 0; frame < num_frams; frame++)
    {
        vecs.push_back(vec);
    }
}

void vecs_2_mat(const std::vector<std::vector<float>> &vecs, std::vector<float> &mat, std::vector<float *> &mem_adds)
{
    uint32_t total_size = 0;
    for (const std::vector<float> &vec : vecs)
    {
        total_size += vec.size();
    }

    mat.reserve(total_size);
    for (const std::vector<float> &vec : vecs)
    {
        mat.insert(mat.end(), vec.begin(), vec.end());
        mem_adds.push_back(&mat[mat.size() - vec.size()]);
    }
}

void compare_vectors(const std::vector<float> &vec1, const std::vector<float> &vec2, float delta)
{
    if (vec1.size() != vec2.size())
    {
        std::cerr << "Error: Vectors are not the same size" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < vec1.size(); i++)
    {
        if (std::abs(vec1[i] - vec2[i]) > delta)
        {
            std::cerr << "Error: Difference between elements at index " << i << " is greater than delta" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

int main()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(3); // Use N threads for all consecutive parallel regions

    const float alpha = 1.0f;
    const float beta = 0.0f;

    uint32_t input_vec_size = 2048;
    uint32_t num_frams = 10;
    uint32_t num_classes = 1000;

    std::vector<float> bias_vec, input_vec, weight_mat, output_vec;

    vector_populator("tensors/example_3/input_vec.txt", input_vec);
    vector_populator("tensors/example_3/weight_mat.txt", weight_mat);
    vector_populator("tensors/example_3/bias_vec.txt", bias_vec);
    vector_populator("tensors/example_3/output_vec.txt", output_vec);

    std::vector<std::vector<float>> bias_vecs, input_vecs, output_vecs;

    vec_2_dup_vecs(bias_vec, bias_vecs, num_frams);
    vec_2_dup_vecs(input_vec, input_vecs, num_frams);   // same inputs for all frams
    vec_2_dup_vecs(output_vec, output_vecs, num_frams); // same inputs for all frams

    std::vector<float> bias_mat, bias_mat_1, bias_mat_2, input_mat, output_mat;
    std::vector<float *> bias_mat_mem_adds, bias_mat_mem_adds_1, bias_mat_mem_adds_2, input_mat_mem_adds, output_mat_mem_adds;

    vecs_2_mat(bias_vecs, bias_mat, bias_mat_mem_adds);
    vecs_2_mat(bias_vecs, bias_mat_1, bias_mat_mem_adds_1);
    vecs_2_mat(bias_vecs, bias_mat_2, bias_mat_mem_adds_2);
    vecs_2_mat(input_vecs, input_mat, input_mat_mem_adds);
    vecs_2_mat(output_vecs, output_mat, output_mat_mem_adds);

    for (int i = 0; i < 10; i++)
    {
        bias_mat_1.clear();
        bias_mat_2.clear();
        std::copy(bias_mat.begin(), bias_mat.end(), std::back_inserter(bias_mat_1));
        std::copy(bias_mat.begin(), bias_mat.end(), std::back_inserter(bias_mat_2));

        Timer::Get().start("sgemv");
        for (uint32_t frame = 0; frame < num_frams; frame++)
        {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_mat_mem_adds[frame], 1, beta, bias_mat_mem_adds_1[frame], 1);
        }
        Timer::Get().stop();

        Timer::Get().start("sgemm");
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_classes, num_frams, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_mat.data(), input_vec_size, beta, bias_mat_2.data(), num_classes);
        Timer::Get().stop();

        compare_vectors(output_mat, bias_mat_1, 0.0001);
        compare_vectors(output_mat, bias_mat_2, 0.0001);

        std::cout<< i << std::endl;  
    }

    Timer::Get().print_duration();

    return 0;
}