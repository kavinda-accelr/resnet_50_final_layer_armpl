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

int main()
{
    // w(number of classes x LLOS) x Trans_i(LLOS x frams) -> o(number of classes x frams)

    // R x C
    // i_vec_size = 3, num frams = 2, num classes = 4
    // i_vec_1 = {13, 14, 15}, i_vec_2 = {16, 17, 18}
    float w[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; 
    float i[] = {13, 14, 15, 16, 17, 18};                
    float b[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};
    
    float b_test[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};

    print_arr(w, 4, 3);
    print_arr(i, 2, 3);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 4, 2, 3, 1.0f, w, 3, i, 3, 1.0f, b, 4);

    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4, 3, 1.0f, w, 3, i, 1, 1.0f, b_test, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4, 3, 1.0f, w, 3, i + 3, 1, 1.0f, b_test + 4, 1);

    print_arr(b, 2, 4);
    print_arr(b_test, 2, 4);
    
    /*
    |   86 104 |
    |  212 257 |
    |  338 410 |
    |  464 563 |

    |         86       212       338       464 |
    |        104       257       410       563 |
    */

    return 0;
}