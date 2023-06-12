#include <armpl.h>
#include <cblas.h>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <omp.h>
#include <cstring>
#include <iostream>
#include <string>

#include "Timer.hpp"

typedef float tensor_t;

class Example_Mats
{
public:
    tensor_t *A, *B, *C;
    const int m, n, k;

    Example_Mats(const int m, const int n, const int k)
    :m(m), n(n), k(k)
    {
        A = new tensor_t[m * k];
        B = new tensor_t[k * n];
        C = new tensor_t[m * n];
    }
    ~Example_Mats()
    {
        delete[] A;
        delete[] B;
        delete[] C;
    }
    void Fill(const int seed=0)
    {
        srand(seed);

        // Fill matrices with random numbers
        for (int i = 0; i < m * k; i++)
        {
            A[i] = (tensor_t)(rand() % 100) / (tensor_t)((rand() % 100) + 0.001);
        }
        for (int i = 0; i < k * n; i++)
        {
            B[i] = (tensor_t)(rand() % 100) / (tensor_t)((rand() % 100) + 0.001);
        }
        for (int i = 0; i < m * n; i++)
        {
            C[i] = (tensor_t)(rand() % 100) / (tensor_t)((rand() % 100) + 0.001);
        }
    }
};

void print_mat(tensor_t *mat, const int m, const int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void arm_pl_gemm(const double alpha, const double beta, const int m, const int n, const int k, tensor_t *R)
{
    Example_Mats Mats(m, n, k);
    Mats.Fill(0);

    tensor_t* C = new tensor_t[m * n];

    Timer::Get().start("ARM");
    memcpy((void *)C, (void *)Mats.C, m * n * sizeof(tensor_t));
    // Perform matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha, Mats.A, k,
                Mats.B, n,
                beta, C, n);
    
    Timer::Get().stop(); 

    memcpy((void *)R, (void *)C, m * n * sizeof(tensor_t));
}

void gemm(const double alpha, const double beta, const int m, const int n, const int k, tensor_t *R)
{
    Example_Mats Mats(m, n, k);
    Mats.Fill(0);

    Timer::Get().start("Loop");
    tensor_t temp = 0;
    // Perform matrix multiplication and addition
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            temp = 0;
            for (int inner = 0; inner < k; inner++)
            {
                temp += Mats.A[row * k + inner] * Mats.B[inner * n + col];
            }
            Mats.C[row * n + col] = (temp * alpha) + beta * Mats.C[row * n + col];
        }
    }
    Timer::Get().stop();

    memcpy((void *)R, (void *)Mats.C, m * n * sizeof(tensor_t));
}

bool value_cmp(tensor_t* R1, tensor_t* R2, const int m, const int n, const double delta=0)
{
    double _delta = 0;
    for(int i=0; i<m*n; i++)
    {
        _delta = R1[i] - R2[i];
        _delta = (_delta < 0.0 ? (-1.0 *_delta) : _delta);
        if(_delta > delta) return false;
    }

    return true;
}

int main()
{

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(1); // Use 4 threads for all consecutive parallel regions

    // (1 x 2048) x (2048 x 1000) -> (1 x 1000)
    // (m x   k ) x (  k  x  n  ) -> (m x   n )

    // Set matrix dimensions
    const int m = 1;
    const int n = 1000;
    const int k = 2048;

    tensor_t *R1 = new tensor_t[m * n];
    tensor_t *R2 = new tensor_t[m * n];

    const double alpha = 1.2; 
    const double beta = 2.3; 

    arm_pl_gemm(alpha, beta, m, n, k, R1);
    gemm(alpha, beta, m, n, k, R2);

    // print_mat(R1, m, n);
    // print_mat(R2, m, n);

    // bool status = memcmp((void *)R1, (void *)R2, m * n * sizeof(tensor_t));
    std::string status = (value_cmp(R1, R2, m, n, 0.00001) ? "Pass" : "Fail");
    std::cout << status << std::endl;

    delete[] R1;
    delete[] R2;

    Timer::Get().print_duration();

    return 0;
}