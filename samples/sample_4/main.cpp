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
    float weight_mat[] = {1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f};
    float bias_vec[] = {0.1f, 0.2f, 0.3f};
    float x[] = {1.0f, 2.0f};

    // bias_vec := alpha * weight_mat * x + beta * bias_vec
    // (3 x 2) x (2 x 1) = (3 x 1)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 2, 1.0f, weight_mat, 2, x, 1, 1.0f, bias_vec, 1);

    // Print the result
    std::cout << "y = [" << bias_vec[0] << ", " << bias_vec[1] << ", " << bias_vec[2] << "]" << std::endl;


    const std::string samples_path = "./tensors";
    const std::string layer_54_output_file_path = samples_path + "/o_54.txt";
    const std::string layer_55_wight_mat_file_path = samples_path + "/w_55.txt";
    const std::string layer_55_output_file_path = samples_path + "/o_55.txt";

    std::vector<tensor_t> layer_54_output;    // (2048)
    std::vector<tensor_t> layer_55_wight_mat; // (1000 x 2048)
    std::vector<tensor_t> layer_55_biases_vec(1000, 0.0f); // (1000)

    // (1000 x 2048) x (2048 x 1) = (1000 x 1)

    vector_populator(layer_54_output_file_path, layer_54_output);
    vector_populator(layer_55_wight_mat_file_path, layer_55_wight_mat);
    // vector_populator(layer_55_output_file_path, layer_55_output);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, 1000, 2048, alpha, layer_55_wight_mat.data(), 2048, layer_54_output.data(), 1, beta, layer_55_biases_vec.data(), 1);
    
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