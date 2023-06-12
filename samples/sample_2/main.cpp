#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <sstream>
#include <fstream>

#include <armpl.h>
#include <cblas.h>
#include <omp.h>

typedef float tensor_t;

std::vector<int> array_dimensions(const std::string &str)
{
    std::string delimiter_1 = "(";
    std::string delimiter_2 = ")";
    int str_start = str.find(delimiter_1) + 1;
    int str_end = str.find(delimiter_2) - str_start;
    std::string token = str.substr(str_start, str_end);
    token = std::regex_replace(token, std::regex(","), "");

    std::vector<int> arr;
    std::stringstream sstream(token);
    int temp;
    while (sstream >> temp)
        arr.push_back(temp);

    return arr;
}

void vector_populator(const std::string &name, std::vector<tensor_t> &vec)
{
    std::ifstream file(name);
    if (!file.is_open())
    {
        std::cout << "File not opened..\n";
        exit(EXIT_FAILURE);
    }

    std::string str;
    std::vector<std::string> arr;
    while (getline(file, str))
    {
        arr.push_back(str);
    }

    std::vector<int> dim_array = array_dimensions(arr[0]);

    unsigned int size = 1;
    for (int val : dim_array)
    {
        size *= val;
    }

    for (unsigned int i = 0; i < size; i++)
    {
        vec.push_back(std::stof(arr[i + 1]));
    }
}

int main()
{
    const std::string samples_path = "./tensors";
    const std::string layer_54_output_file_path = samples_path + "/o_54.txt";
    const std::string layer_55_wight_mat_file_path = samples_path + "/w_55.txt";
    const std::string layer_55_output_file_path = samples_path + "/o_55.txt";

    std::vector<tensor_t> layer_54_output;    // (2048 x 1)
    std::vector<tensor_t> layer_55_wight_mat; // (1000 x 2048)
    std::vector<tensor_t> layer_55_biases_vec(1000, 0.0f); // (1000 x 1)
    // std::vector<tensor_t> layer_55_output; // (1000 x 1)

    // y = wx = (1000 x 2048) x (2048 x 1)

    vector_populator(layer_54_output_file_path, layer_54_output);
    vector_populator(layer_55_wight_mat_file_path, layer_55_wight_mat);
    // vector_populator(layer_55_output_file_path, layer_55_output);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // dimensions of weight matrix
    const int M = 1000; // number of rows
    const int N = 2048; // number of columns

    // dimensions of input vector
    const int K = 1; // number of columns

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, K, N, alpha,
                layer_55_wight_mat.data(), N,
                layer_54_output.data(), K, beta,
                layer_55_biases_vec.data(), K);

    std::cout << std::endl;

    int i = 0;
    for (auto val : layer_55_biases_vec)
    {
        std::cout << val << std::endl;
        i++;
        if (i == 10)
            break;
    }

    return 0;
}