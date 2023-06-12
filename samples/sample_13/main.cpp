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

void compare_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2, float delta) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Vectors are not the same size" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < vec1.size(); i++) {
        if (std::abs(vec1[i] - vec2[i]) > delta) {
            std::cerr << "Error: Difference between elements at index " << i << " is greater than delta" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

int main()
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /*
    uint32_t input_vec_size = 4;
    uint32_t num_frams = 3;
    uint32_t num_classes = 5;

    const std::vector<float> bias_vec{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    const std::vector<std::vector<float>> input_vecs{{21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}};
    const std::vector<float> weight_mat{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    */
    uint32_t input_vec_size = 2048;
    uint32_t num_frams = 1;
    uint32_t num_classes = 1000;

    std::vector<float> bias_vec;
    std::vector<std::vector<float>> input_vecs;
    std::vector<float> weight_mat;

    input_vecs.push_back({});
    vector_populator("tensors/example_3/input_vec.txt", input_vecs[0]);
    vector_populator("tensors/example_3/weight_mat.txt", weight_mat);
    vector_populator("tensors/example_3/bias_vec.txt", bias_vec);

    std::vector<float> output_vec;
    vector_populator("tensors/example_3/output_vec.txt", output_vec);
    
    std::vector<float> bias_mat;
    std::vector<std::vector<float>> bias_vecs;
    std::vector<float> input_mat;

    setup_bias_mat(bias_vec, bias_mat, num_frams);
    setup_bias_vecs(bias_vec, bias_vecs, num_frams);
    setup_input_mat(input_vecs, input_mat);

    for (uint32_t frame = 0; frame < num_frams; frame++)
    {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_vecs[frame].data(), 1, beta, bias_vecs[frame].data(), 1);
    }

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_classes, num_frams, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_mat.data(), input_vec_size, beta, bias_mat.data(), num_classes);
    
    for(int i=0; i<10; i++)
    {
        std::cout << output_vec[i] << " : " << bias_mat[i] << " : " << bias_vecs[0][i] << std::endl;
    }
    compare_vectors(output_vec, bias_mat, 0.0001);

    return 0;
}