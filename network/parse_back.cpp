#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <map>
#include <bits/stdc++.h> 
#include <boost/algorithm/string.hpp>
#include "conv2D.h"

template<typename T>
using weight_matrices = std::vector<std::vector<std::vector<std::vector<std::vector<T>>>>>;

template<typename T>
using matrix_4d = std::vector<std::vector<std::vector<std::vector<T>>>>;

template<typename T>
using matrix_3d = std::vector<std::vector<std::vector<T>>>;

template<typename T>
using matrix_2d = std::vector<std::vector<T>> ;

template<typename T>
using matrix_1d = std::vector<T>;

int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(std::string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

std::vector<std::string> get_files(std::string name){

    std::string dir = std::string(name);
    std::vector<std::string> files = std::vector<std::string>();
    getdir(dir,files);
    std::vector<int> erase_arr;
    for (unsigned int i = 0;i < files.size();i++) {
        if(std::strcmp(".",files[i].c_str()) == 0 || std::strcmp("..", files[i].c_str())== 0){
            erase_arr.push_back(i);
        }      
    }
    for (int i = erase_arr.size()-1; i >= 0;i--)
    {
        files.erase(files.begin() + erase_arr[i]);
    }

    std::sort(files.begin(), files.end());
    return files;
}

std::vector<float> parse_bias_alpha(std::string path){
    std::ifstream file;
    file.open(path);
    std::string s,name;
    std::getline(file,s);
    std::vector<std::string> stringVector; 
    boost::split(stringVector, s, boost::is_any_of(","));
    stringVector.erase(stringVector.begin() + stringVector.size()-1);
    std::vector<float> floatVector(stringVector.size());

    std::transform(stringVector.begin(), stringVector.end(), floatVector.begin(), [](const std::string& val)
    {
        return std::stof(val);
    });

    return floatVector;
}

template <typename T>
matrix_4d<T> parse_weights(std::string path){

    matrix_4d<T> matrix4d;
    matrix_3d<T> matrix3d;
    matrix_2d<T> matrix2d;
    matrix_1d<T> matrix1d;

    std::ifstream file;
    file.open(path);
    std::string s;
    std::getline(file,s);
    std::vector<std::string> d1;
    std::vector<std::string> d2;
    std::vector<std::string> d3;
    std::vector<std::string> d4;

    boost::split(d1, s, boost::is_any_of("*"));
    d1.erase(d1.begin() + d1.size()-1);
    for (int i =0; i < d1.size(); i++){
        boost::split(d2, d1[i], boost::is_any_of("/"));
        d2.erase(d2.begin() + d2.size()-1);
        for(int j=0; j < d2.size(); j++){
            boost::split(d3, d2[j], boost::is_any_of("_"));
            d3.erase(d3.begin() + d3.size()-1);
            for(int k=0; k<d3.size(); k++){
                boost::split(d4, d3[k], boost::is_any_of(","));
                d4.erase(d4.begin() + d4.size()-1);
                std::transform(d4.begin(), d4.end(), back_inserter(matrix1d), [](const std::string& val)
                {
                    return std::stof(val);
                });     

                matrix2d.push_back(matrix1d);
                matrix1d.clear();
            }
            matrix3d.push_back(matrix2d);
            matrix2d.clear();
        }
        matrix4d.push_back(matrix3d);
        matrix3d.clear();
    }

    return matrix4d;

}

int main()
{
    std::string b_path = "bias/";
    std::string a_path = "alpha/";
    std::string w_path = "weights/";

    // std::map<std::string,std::vector<float>> biases;
    // std::map<std::string,std::vector<float>> alphas;

    matrix_2d<float> biases2;
    matrix_2d<float> alphas2;

    std::vector<std::string> b_files = get_files("bias");

    for (int i = 0; i < b_files.size(); i++)
    {
        std::vector<float> value = parse_bias_alpha(b_path + b_files[i]);
        biases2.push_back(value);
    }

    // for (int i = 0; i < b_files.size(); i++)
    // {
    //     std::vector<float> value = parse_bias_alpha(b_path + b_files[i]);
    //     biases.insert(std::pair<std::string,std::vector<float>>(b_files[i],value));
    // }

    std::vector<std::string> a_files = get_files("alpha");

    for (int i = 0; i < a_files.size(); i++)
    {
        std::vector<float> value = parse_bias_alpha(a_path + a_files[i]);
        alphas2.push_back(value);
    }

    // for (int i = 0; i < a_files.size(); i++)
    // {
    //     std::vector<float> value = parse_bias_alpha(a_path + a_files[i]);
    //     alphas.insert(std::pair<std::string,std::vector<float>>(a_files[i],value));
    // }

    std::vector<std::string> w_files = get_files("weights");

    weight_matrices<float> in_out_weight;
    weight_matrices<float> hidden_weights;
    matrix_4d<float> f_matrix4d;
    matrix_4d<float> i_matrix4d;

    for (int i = 0; i < w_files.size(); i++)
    {

        if( i==0 || i == (w_files.size()-1) ){
            f_matrix4d = parse_weights<float>(w_path + w_files[i]);
            in_out_weight.push_back(f_matrix4d);
        }

        else{
            i_matrix4d = parse_weights<float>(w_path + w_files[i]);
            hidden_weights.push_back(i_matrix4d);
        }

    }

    // out, in, height, width;

    matrix_3d<float> input_m = hidden_weights[0][0];
    conv2D<float>(input_m, f_matrix4d);
    return 0;

}