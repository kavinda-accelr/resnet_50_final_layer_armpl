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

template <typename T>
void print_arr(T *arr, size_t rows, size_t columns)
{
    for (size_t i = 0; i < rows; i++)
    {
        std::cout << " | ";
        for (size_t j = 0; j < columns; j++)
        {
            std::cout << std::setw(15) << (float)arr[(columns * i) + j];
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

    mat.clear();
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
            std::cerr << "Error: Difference between elements at index " << i << " is greater than delta ["<< vec1[i] << " - " << vec2[i] <<"]"<< std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

void text_samples()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(3); // Use N threads for all consecutive parallel regions

    const float alpha = 1.0f;
    const float beta = 0.0f;

    uint32_t input_vec_size = 2048;
    uint32_t num_frams = 1;
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

        std::cout << i << std::endl;
    }

    Timer::Get().print_duration();
}

void single_frame_benchmark()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(3); // Use N threads for all consecutive parallel regions

    const float alpha = 1.0f;
    const float beta = 0.0f;

    uint32_t input_vec_size = 2048;
    uint32_t num_classes = 1000;

    std::vector<float> bias_vec, bias_vec_results, input_vec, weight_mat, output_vec;

    vector_populator("tensors/example_3/input_vec.txt", input_vec);
    vector_populator("tensors/example_3/weight_mat.txt", weight_mat);
    vector_populator("tensors/example_3/bias_vec.txt", bias_vec);
    vector_populator("tensors/example_3/output_vec.txt", output_vec);

    const int cycles = 100;
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        bias_vec_results.clear();
        std::copy(bias_vec.begin(), bias_vec.end(), std::back_inserter(bias_vec_results));

        Timer::Get().start("sgemv");
        cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_vec.data(), 1, beta, bias_vec_results.data(), 1);
        Timer::Get().stop();

        compare_vectors(output_vec, bias_vec_results, 0.000001);

        std::cout << cycle << std::endl;
    }

    Timer::Get().print_duration();

    /*
    Time in ms
    Block name          CPU TIME USED       PROCESS_CPUTIME_ID  MONOTONIC           MONOTONIC_RAW       REALTIME            Cycles
    sgemv               20.5468             20.5418             6.85441             6.85426             6.85454             100
    */
}

template <typename T>
void fill_int_vec(std::vector<T> &vec)
{
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (T &val : vec)
    {
        val = dis(gen);
    }
}

void fill_float_vec(std::vector<float> &vec)
{
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;
    for (float &val : vec)
    {
        val = dis(gen) - 0.5f;
    }
}

void benchmark_mat()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(3); // Use N threads for all consecutive parallel regions

    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;
    const float out_scale = dis(gen) - 0.5f;
    const float alpha = 1.0f;
    const float beta = 1.0f;

    uint32_t input_vec_size = 2048;
    uint32_t num_frams = 80;
    uint32_t num_classes = 1000;

    uint32_t input_mat_size = input_vec_size * num_frams;
    uint32_t weight_mat_size = input_vec_size * num_classes;

    std::vector<float> bias_vec(num_classes), bias_mat, bias_mat_results, input_mat(input_mat_size), weight_mat(weight_mat_size);
    std::vector<std::vector<float>> bias_vecs;
    std::vector<float *> bias_mat_mem_adds, bias_mat_results_adds;
    std::vector<int8_t> input_int_mat(input_mat_size);
    std::vector<uint32_t> argmax_results;

    fill_float_vec(weight_mat);
    fill_float_vec(bias_vec);
    vec_2_dup_vecs(bias_vec, bias_vecs, num_frams);
    vecs_2_mat(bias_vecs, bias_mat, bias_mat_mem_adds);
    vecs_2_mat(bias_vecs, bias_mat_results, bias_mat_results_adds);

    int cycles = 100;
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        fill_int_vec(input_int_mat);

        Timer::Get().start("ALL");

        // Timer::Get().start("transform");
        std::transform(input_int_mat.data(), input_int_mat.data() + input_int_mat.size(), input_mat.data(), [](int8_t x)
                       { return static_cast<float>(x); });
        // Timer::Get().stop();

        // Timer::Get().start("scale");
        cblas_sscal(input_mat.size(), out_scale, input_mat.data(), 1);
        // Timer::Get().stop();

        // Timer::Get().start("copy");
        std::copy(bias_mat.data(), bias_mat.data() + bias_mat.size(), bias_mat_results.data());
        // Timer::Get().stop();

        // Timer::Get().start("sgemm");
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_classes, num_frams, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_mat.data(), input_vec_size, beta, bias_mat_results.data(), num_classes);
        // Timer::Get().stop();

        // Timer::Get().start("argmax");
        for (uint32_t frame = 0; frame < num_frams; frame++)
        {
            float *max_it = std::max_element(bias_mat_results_adds[frame], bias_mat_results_adds[frame] + num_classes);
            int max_index = std::distance(bias_mat_results_adds[frame], max_it);
            argmax_results.push_back(max_index);
        }
        // Timer::Get().stop();

        Timer::Get().stop();

        std::cout << cycle << std::endl;
    }

    std::cout << argmax_results << std::endl;
    Timer::Get().print_duration();

    /*
    1400
    Time in ms
    Block name          CPU TIME USED       PROCESS_CPUTIME_ID  MONOTONIC           MONOTONIC_RAW       REALTIME            Cycles
    ALL                 2026.87             2026.87             693.054             693.024             693.054             100

    10
    Time in ms
    Block name          CPU TIME USED       PROCESS_CPUTIME_ID  MONOTONIC           MONOTONIC_RAW       REALTIME            Cycles
    ALL                 93.3457             93.3416             32.5242             32.5236             32.5242             100

    80 frams per 57.8655 ms -> 0.725 ms per frams -> 1379.31 FPS
    Time in ms
    Block name          CPU TIME USED       PROCESS_CPUTIME_ID  MONOTONIC           MONOTONIC_RAW       REALTIME            Cycles
    ALL                 168.48              168.476             57.8653             57.8642             57.8655             100
    */
}

void benchmark_vec()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(3); // Use N threads for all consecutive parallel regions

    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis;
    const float out_scale = dis(gen) - 0.5f;
    const float alpha = 1.0f;
    const float beta = 1.0f;

    uint32_t input_vec_size = 2048;
    uint32_t num_classes = 1000;

    uint32_t weight_mat_size = input_vec_size * num_classes;

    std::vector<float> bias_vec(num_classes), bias_vec_results(num_classes), weight_mat(weight_mat_size), input_vec(input_vec_size);
    std::vector<int8_t> input_int_vec(input_vec_size);
    std::vector<uint32_t> argmax_results;

    fill_float_vec(weight_mat);
    fill_float_vec(bias_vec);

    int cycles = 100;
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        fill_int_vec(input_int_vec);

        Timer::Get().start("ALL");

        // Timer::Get().start("transform");
        std::transform(input_int_vec.data(), input_int_vec.data() + input_int_vec.size(), input_vec.data(), [](int8_t x)
                       { return static_cast<float>(x); });
        // Timer::Get().stop();

        // Timer::Get().start("scale");
        cblas_sscal(input_vec.size(), out_scale, input_vec.data(), 1);
        // Timer::Get().stop();

        // Timer::Get().start("copy");
        std::copy(bias_vec.data(), bias_vec.data() + bias_vec.size(), bias_vec_results.data());
        // Timer::Get().stop();

        // Timer::Get().start("sgemm");
        cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_vec.data(), 1, beta, bias_vec_results.data(), 1);
        // Timer::Get().stop();

        // Timer::Get().start("argmax");
        auto max_it = std::max_element(bias_vec_results.begin(), bias_vec_results.end());
        int max_index = std::distance(bias_vec_results.begin(), max_it);
        argmax_results.push_back(max_index);
        // Timer::Get().stop();

        Timer::Get().stop();

        std::cout << cycle << std::endl;
    }

    std::cout << argmax_results << std::endl;
    Timer::Get().print_duration();
}

int spmat()
{
    const uint32_t M = 2;
    const uint32_t N = 2;
    /* 1. Set-up local CSR structure */
    armpl_spmat_t armpl_mat_a, armpl_mat_b, armpl_mat_c;
    armpl_status_t info;
    const double alpha = 1.0, beta = 1.0;

    double mat_a[M * N] = {1.0, 2.0, 3.0, 4.0};
    double mat_b[N * M] = {1.0, 2.0, 3.0, 4.0};
    double mat_c[M * M] = {1.0, 2.0, 3.0, 4.0};

    /*
    1 2    1 2
    3 4  X 3 4 =
    */

    info = armpl_spmat_create_dense_d(&armpl_mat_a, ARMPL_ROW_MAJOR, M, N, M, mat_a, 0);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmat_create_dense_d returned %d\n", info);

    info = armpl_spmat_create_dense_d(&armpl_mat_b, ARMPL_ROW_MAJOR, N, M, N, mat_b, 0);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmat_create_dense_d returned %d\n", info);

    info = armpl_spmat_create_dense_d(&armpl_mat_c, ARMPL_ROW_MAJOR, M, M, M, mat_c, 0);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmat_create_dense_d returned %d\n", info);

    info = armpl_spmv_optimize(armpl_mat_a);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmv_optimize returned %d\n", info);

    info = armpl_spmv_optimize(armpl_mat_b);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmv_optimize returned %d\n", info);

    info = armpl_spmv_optimize(armpl_mat_c);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmv_optimize returned %d\n", info);

    info = armpl_spmm_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, ARMPL_SPARSE_OPERATION_NOTRANS,
                             alpha, armpl_mat_a, armpl_mat_b, beta, armpl_mat_c);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmm_exec_d returned %d\n", info);

    armpl_int_t ARM_M = 2;
    double row_1[M];
    double row_2[M];
    double *results[M] = {row_1, row_2};
    info = armpl_spmat_export_dense_d(armpl_mat_c, ARMPL_ROW_MAJOR, &ARM_M, &ARM_M, results);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmat_export_dense_d returned %d\n", info);

    printf("Computed mat c :\n");
    for (uint32_t i = 0; i < M; i++)
    {
        printf("\t%2.1f\n", row_1[i]);
    }

    for (uint32_t i = 0; i < M; i++)
    {
        printf("\t%2.1f\n", row_2[i]);
    }

    info = armpl_spmat_destroy(armpl_mat_a);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmat_destroy returned %d\n", info);

    info = armpl_spmat_destroy(armpl_mat_b);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmat_destroy returned %d\n", info);

    info = armpl_spmat_destroy(armpl_mat_c);
    if (info != ARMPL_STATUS_SUCCESS)
        printf("ERROR: armpl_spmat_destroy returned %d\n", info);

    return (int)info;
}

int main()
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(3); // Use N threads for all consecutive parallel regions

    const float alpha = 1.0f;
    const float beta = 1.0f;

    uint32_t input_vec_size = 2048;
    uint32_t num_classes = 1000;

    std::vector<float> bias_vec, bias_vec_results, input_vec, weight_mat, output_vec;

    vector_populator("tensors/example_4/input_vec.txt", input_vec);
    vector_populator("tensors/example_4/weight_mat.txt", weight_mat);
    vector_populator("tensors/example_4/bias_vec.txt", bias_vec);
    vector_populator("tensors/example_4/output_vec.txt", output_vec);

    const int cycles = 100;
    for (int cycle = 0; cycle < cycles; cycle++)
    {
        bias_vec_results.clear();
        std::copy(bias_vec.begin(), bias_vec.end(), std::back_inserter(bias_vec_results));

        Timer::Get().start("sgemv");
        cblas_sgemv(CblasRowMajor, CblasNoTrans, num_classes, input_vec_size, alpha, weight_mat.data(), input_vec_size, input_vec.data(), 1, beta, bias_vec_results.data(), 1);
        Timer::Get().stop();

        compare_vectors(output_vec, bias_vec_results, 0.00001);

        std::cout << cycle << std::endl;
    }

    Timer::Get().print_duration();

    return 0;
}