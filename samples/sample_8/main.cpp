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

void rm_test()
{
    // flag RM
    // w RM, i CM, o RM
    // w(number of classes x LLOS) x Trans_i(LLOS x frams) -> o(number of classes x frams)

    // flag CM
    // w CM, i CM, o CM
    // Trans_w(number of classes x LLOS) x i(LLOS x frams) -> o(number of classes x frams)

    // R x C
    // i_vec_size = 3, num frams = 2, num classes = 4
    // i_f1 = {13, 14, 15}, i_f2 = {16, 17, 18}
    float w[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // 4 x 3
    float i[] = {13, 14, 15, 16, 17, 18};                // 2 x 3 - op(i) 3 x 2
    float b[] = {0, 0, 0, 0, 0, 0, 0, 0};                // 4 x 2

    print_arr(w, 4, 3);
    print_arr(i, 2, 3);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 4, 2, 3, 1.0f, w, 3, i, 3, 0.0f, b, 2);

    print_arr(b, 4, 2);
    /*
    |   86 104 |
    |  212 257 |
    |  338 410 |
    |  464 563 |
    */
}

void cm_test_notransw()
{
    // flag RM
    // w RM, i CM, o RM
    // w(number of classes x LLOS) x Trans_i(LLOS x frams) -> o(number of classes x frams)

    // flag CM
    // w CM, i CM, o CM
    // Trans_w(number of classes x LLOS) x i(LLOS x frams) -> o(number of classes x frams)

    // R x C
    // i_vec_size = 3, num frams = 2, num classes = 4
    // i_f1 = {13, 14, 15}, i_f2 = {16, 17, 18}
    // float w[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // 4 x 3
    float w[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12}; // 4 x 3
    float i[] = {13, 14, 15, 16, 17, 18};                // 2 x 3 - op(i) 3 x 2
    float b[] = {0, 0, 0, 0, 0, 0, 0, 0};                // 4 x 2

    print_arr(w, 4, 3);
    print_arr(i, 2, 3);

    // 4 x 3 * 3 * 2 -> 4 x 2
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 2, 3, 1.0f, w, 4, i, 3, 0.0f, b, 4);

    print_arr(b, 2, 4);
    /*
    |   86 104 |
    |  212 257 |
    |  338 410 |
    |  464 563 |

    |         86       212       338       464 |
    |        104       257       410       563 |
    */
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

    print_arr(w, 4, 3);
    print_arr(i, 2, 3);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 4, 2, 3, 1.0f, w, 3, i, 3, 1.0f, b, 4);

    print_arr(b, 2, 4);
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