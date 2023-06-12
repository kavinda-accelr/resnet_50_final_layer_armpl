#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>

#include <armpl.h>
#include <cblas.h>
#include <omp.h>

#include "Timer.h"

double get_max_diff_vecs(const std::vector<float> &vec_1, const std::vector<float> &vec_2)
{
    double delta = 0;
    double delta_max = 0;
    if (vec_1.size() != vec_2.size())
    {
        std::cout << "Size mismatch\n";
        return -1;
    }

    for (size_t i = 0; i < vec_1.size(); i++)
    {
        delta = vec_1[i] - vec_2[i];
        delta = (delta < 0.0 ? (-1.0 * delta) : delta);
        delta_max = (delta_max < delta ? delta : delta_max);
    }

    return delta_max;
}

void gemv(
    const std::vector<float> &matrix,
    const std::vector<float> &vector,
    const std::vector<float> &bias,
    std::vector<float> &result_vector,
    int rows,
    int cols)
{
    for (int i = 0; i < rows; i++)
    {
        result_vector[i] = bias[i];
        for (int j = 0; j < cols; j++)
        {
            result_vector[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

template <typename T, size_t SIZE>
void print_arr(const std::array<T, SIZE> &arr, uint32_t items = SIZE)
{
    uint32_t i = 0;
    for (const T &val : arr)
    {
        std::cout << (float)val << std::endl;
        i++;
        if (i == items)
            break;
    }
    std::cout << std::endl;
}

template <typename T>
void print_arr(const std::vector<T> &arr, uint32_t items)
{
    uint32_t i = 0;
    for (const T &val : arr)
    {
        std::cout << (float)val << std::endl;
        i++;
        if (i == items)
            break;
    }
    std::cout << std::endl;
}

template <typename T>
void print_arrs(const std::vector<T> &arr_1, const std::vector<T> &arr_2, int items)
{
    for (int i = 0; i < items; i++)
    {
        std::cout << (float)arr_1[i] << " : " << (float)arr_2[i] << std::endl;
    }
    std::cout << std::endl;
}

template <size_t SIZE>
void fill_float_arr(std::array<float, SIZE> &arr)
{
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;
    for (float &val : arr)
    {
        val = dis(gen) - 0.5f;
    }
}

void fill_float_arr(std::vector<float> &arr)
{
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;
    for (float &val : arr)
    {
        val = dis(gen) - 0.5f;
    }
}

template <typename T, size_t SIZE>
void fill_int_arr(std::array<T, SIZE> &arr)
{
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (T &val : arr)
    {
        val = dis(gen);
    }
}

template <typename T>
void fill_int_arr(std::vector<T> &arr)
{
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (T &val : arr)
    {
        val = dis(gen);
    }
}

int main()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(3); // Use N threads for all consecutive parallel regions

    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const float out_scale = dis(gen) - 0.5f;

    const uint32_t input_vec_size = 2048;
    const uint32_t biases_vec_size = 1000;
    const uint32_t wight_mat_rows = biases_vec_size;
    const uint32_t wight_mat_cols = input_vec_size;

    const int num_frams = 20;
    std::vector<int8_t> input_vec_int8(input_vec_size * num_frams);          // (2048 x num_frams)
    std::vector<float> input_vec(input_vec_size * num_frams);                // (2048 x num_frams)
    std::vector<float> wight_mat(wight_mat_rows * wight_mat_cols);           // (1000 x 2048)
    std::vector<float> biases_vec(biases_vec_size * num_frams);              // (1000 x num_frams)
    std::vector<float> biases_vec_for_results(biases_vec_size * num_frams);  // (1000 x num_frams)

    fill_float_arr(biases_vec);
    fill_float_arr(wight_mat);

    /*
    Timer::Get().start("timer check");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    Timer::Get().stop();
    */

    std::vector<int> argmax_results;
    int cycles = 1000;
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        fill_int_arr(input_vec_int8);

        Timer::Get().start("transform");
        std::transform(input_vec_int8.data(), input_vec_int8.data() + input_vec_int8.size(), input_vec.data(), [](int8_t x)
                       { return static_cast<float>(x); });
        Timer::Get().stop();
        Timer::Get().start("scale");
        cblas_sscal(input_vec.size(), out_scale, input_vec.data(), 1);
        Timer::Get().stop();

        Timer::Get().start("copy");
        std::copy(biases_vec.data(), biases_vec.data() + biases_vec.size(), biases_vec_for_results.data());
        Timer::Get().stop();

        Timer::Get().start("gemm");
        // cblas_sgemv(CblasRowMajor, CblasNoTrans, wight_mat_rows, wight_mat_cols, alpha, wight_mat.data(), wight_mat_cols, input_vec.data(), 1, beta, biases_vec_for_results.data(), 1); // 1.8 ms -> 20T
        
        const int m = 1000;
        const int n = num_frams;
        const int k = 2048;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha, wight_mat.data(), k,
                input_vec.data(), n,
                beta, biases_vec_for_results.data(), n);
        Timer::Get().stop();

        // Timer::Get().start("argmax");
        // auto max_it = std::max_element(biases_vec_for_results.begin(), biases_vec_for_results.end());
        // int max_index = std::distance(biases_vec_for_results.begin(), max_it);
        // Timer::Get().stop();

        // argmax_results.push_back(max_index);
    }

    // std::cout << "cycles : " << argmax_results.size() << std::endl;
    Timer::Get().print_duration();

    return 0;
}