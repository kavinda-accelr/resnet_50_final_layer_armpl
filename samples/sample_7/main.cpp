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
        std::cout << arr_1[i] << " : " << arr_2[i] << std::endl;
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
    // std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dis;
    for (float &val : arr)
    {
        val = dis(gen) - 0.5f;
    }
}

void fill_float_buffer(float* ptr, size_t size)
{
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;
    for(size_t i = 0; i<size; i++)
    {
        ptr[i] = dis(gen) - 0.5f;
    }
}

template <typename T, size_t SIZE>
void fill_int_arr(std::array<T, SIZE> &arr)
{
    // std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (T &val : arr)
    {
        val = dis(gen);
    }
}

template <typename T>
void fill_int_arr(std::vector<T> &arr)
{
    // std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (T &val : arr)
    {
        val = dis(gen);
    }
}

int main()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(1); // Use N threads for all consecutive parallel regions

    // std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dis;

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const float out_scale = dis(gen) - 0.5f;

    const uint32_t input_vec_size = 3;
    const uint32_t output_vec_size = 4;
    const uint32_t num_frams = 1;

    // MAT B**T = op(B) : Row Major
    const uint32_t input_mat_columns = num_frams;
    const uint32_t input_mat_rows = input_vec_size;

    const uint32_t biases_mat_columns = num_frams;
    const uint32_t biases_mat_rows = output_vec_size;
    
    // MAT A = op(A) : Row Major
    const uint32_t wight_mat_rows = output_vec_size;
    const uint32_t wight_mat_columns = input_vec_size;

    std::vector<int8_t> input_mat_int8(input_mat_columns * input_mat_rows);                    
    std::vector<float> input_mat(input_mat_columns * input_mat_rows);                          
    std::vector<float> wight_mat(wight_mat_columns * wight_mat_rows);                          
    // std::vector<float> biases_mat(biases_mat_columns * biases_mat_rows);                  
    // std::vector<float> biases_mat_for_results(biases_mat_columns * biases_mat_rows);      
    // std::vector<float> biases_mat_for_results_test(biases_mat_columns * biases_mat_rows); 
    std::vector<float> biases_mat_for_results(biases_mat_columns * biases_mat_rows);
    std::vector<float> biases_mat_for_results_test(biases_mat_columns * biases_mat_rows);

    // fill_float_buffer(biases_mat.data(), biases_mat_columns);
    // for(uint32_t i=1; i<input_mat_rows;i++)
    // {
    //     std::copy(biases_mat.data(), biases_mat.data() + biases_mat_columns, biases_mat.data() + (i*biases_mat_columns));
    // }
    
    fill_float_arr(wight_mat);

    /*
    Timer::Get().start("timer check");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    Timer::Get().stop();
    */

    std::vector<int> argmax_results;
    int cycles = 1;
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        fill_int_arr(input_mat_int8);

        Timer::Get().start("transform");
        std::transform(input_mat_int8.data(), input_mat_int8.data() + input_mat_int8.size(), input_mat.data(), [](int8_t x)
                       { return static_cast<float>(x); });
        Timer::Get().stop();
        Timer::Get().start("scale");
        cblas_sscal(input_mat.size(), out_scale, input_mat.data(), 1);
        Timer::Get().stop();

        /*
        Timer::Get().start("copy");
        std::copy(biases_mat.data(), biases_mat.data() + biases_mat.size(), biases_mat_for_results.data());
        // biases_mat_for_results = biases_mat;
        Timer::Get().stop();
        std::copy(biases_mat.data(), biases_mat.data() + biases_mat.size(), biases_mat_for_results_test.data());
        // biases_mat_for_results_test = biases_mat;
        */
        // print_arrs(biases_mat_for_results_test, biases_mat_for_results, biases_mat_columns * biases_mat_rows);

        cblas_sgemv(CblasRowMajor, CblasNoTrans, wight_mat_rows, wight_mat_columns, alpha, wight_mat.data(), wight_mat_columns, input_mat.data(), 1, beta, biases_mat_for_results_test.data(), 1); // 1.8 ms -> 20T

        print_arrs(biases_mat_for_results_test, biases_mat_for_results, biases_mat_columns * biases_mat_rows);

        Timer::Get().start("gemm");
        
        const uint32_t m = wight_mat_rows;
        const uint32_t n = input_mat_columns;
        const uint32_t k = wight_mat_columns;

        const uint32_t lda = wight_mat_columns;
        const uint32_t ldb = input_mat_rows;
        const uint32_t ldc = biases_mat_columns;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                alpha, wight_mat.data(), lda,
                input_mat.data(), ldb,
                beta, biases_mat_for_results.data(), ldc);
        Timer::Get().stop();

        print_arrs(biases_mat_for_results_test, biases_mat_for_results, biases_mat_columns * biases_mat_rows);

        // Timer::Get().start("argmax");
        // auto max_it = std::max_element(biases_vec_for_results.begin(), biases_vec_for_results.end());
        // int max_index = std::distance(biases_vec_for_results.begin(), max_it);
        // Timer::Get().stop();

        // argmax_results.push_back(max_index);
    }

    // std::cout << "cycles : " << argmax_results.size() << std::endl;
    // Timer::Get().print_duration();

    return 0;
}