#include <iostream>
#include <array>
#include <random>
#include <chrono>
#include <algorithm>

#include <armpl.h>
#include <cblas.h>
#include <omp.h>

#include "Timer.hpp"

void matrixVectorMultiplication(
    const std::vector<std::vector<float>> &matrix,
    const std::vector<float> &vector,
    std::vector<float> &result_vector)
{
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < vector.size(); j++)
        {
            result_vector[i] += matrix[i][j] * vector[j];
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

int main()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(1); // Use 4 threads for all consecutive parallel regions

    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const float out_scale = dis(gen) - 0.5f;

    const uint32_t input_vec_size = 2048;
    const uint32_t biases_vec_size = 1000;
    const uint32_t wight_mat_rows = biases_vec_size;
    const uint32_t wight_mat_cols = input_vec_size;

    std::array<int8_t, input_vec_size> input_vec_int8;            // (2048)
    std::array<float, input_vec_size> input_vec;                  // (2048)
    std::array<float, wight_mat_rows * wight_mat_cols> wight_mat; // (1000 x 2048)
    std::array<float, biases_vec_size> biases_vec;                // (1000)
    std::array<float, biases_vec_size> biases_vec_for_results;    // (1000)

    fill_float_arr(biases_vec);
    fill_float_arr(wight_mat);

    int cycles = 1000;
    for (int i = 0; i < cycles; i++)
    {
        fill_int_arr(input_vec_int8);

        // Timer::Get().start("Total");

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
        Timer::Get().start("gemv");
        // (1000 x 2048) x (2048 x 1) = (1000 x 1)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, wight_mat_rows, wight_mat_cols, alpha, wight_mat.data(), wight_mat_cols, input_vec.data(), 1, beta, biases_vec_for_results.data(), 1);
        Timer::Get().stop();

        Timer::Get().start("argmax");
        auto max_it = std::max_element(biases_vec_for_results.begin(), biases_vec_for_results.end());
        int max_index = std::distance(biases_vec_for_results.begin(), max_it);
        Timer::Get().stop();

        // Timer::Get().stop();
        std::cout << max_index << std::endl;
    }

    Timer::Get().print_duration();

    return 0;
}