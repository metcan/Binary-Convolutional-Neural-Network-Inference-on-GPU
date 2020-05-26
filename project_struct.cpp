#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <boost/thread.hpp>

using namespace boost::thread;

template <typename T>
class Matrix {
    int col_, row_;
    typedef std::vector<T> Row;
    
    std::vector<Row> data;

public:
    Matrix(int c, int r): row_(r), col_(c), data(c, std::vector<T>(r)){}
    const int& col = col_;
    const int& row = row_;

    Row & operator[](int i)
    {
        return data[i];
    }
};

template <typename T>
class Tensor{
    int col_, row_, channel_;
    typedef Matrix<T> Mat;

    std::vector<Mat> tensor;

public:
    Tensor(int col, int row, int channel): col_(col), row_(row), channel_(channel), tensor(channel, Matrix<T>(col, row)) {}
    const int& col = col_;
    const int& row = row_;
    const int& channel = channel_;

    Mat & operator[](int c)
    {
        return tensor[c];
    }
    std::tuple<int, int, int> get_shape(void)
    {
        return std::make_tuple(channel_, row_, col_);   
    } 
    
};

template <typename T>
void MatrixSquare(Matrix<int> &mat)
{
    for(int j=0; mat.col> j; j++)
    {
        for(int i=0; mat.row > i; i++)
        {
            mat[j][i] = mat[j][i] * mat[j][i];
        }
    }
}

int main()
{
    Matrix<int> matrix(5, 5);
    std::cout<<matrix[0][0]<<std::endl;

    Tensor<int> tensor(256, 256, 8);
    for(int k=0; tensor.channel> k; k++)
    {
        for(int j=0; tensor.col> j; j++)
        {
            for(int i=0; tensor.row > i; i++)
            {
                tensor[k][j][i] = (std::rand()%1000 + 1);
            }
        }
    }
    // SINGLE CORE PERFORMANCE
    auto start = std::chrono::high_resolution_clock::now();
    for(int k=0; tensor.channel> k; k++)
    {
        for(int j=0; tensor.col> j; j++)
        {
            for(int i=0; tensor.row > i; i++)
            {
                tensor[k][j][i] = tensor[k][j][i] * tensor[k][j][i];
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto single_core_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // MULTI THREAD PERFORMANCE
    thread *threads[8];
    for(int k=0; tensor.channel> k; k++)
    {
        // ThreadVector.emplace_back([&])(){MatrixSquare<int>(tensor[k])}
        threads[k] = new thread(MatrixSquare<int>, tensor[k]); 
    }
    for(int k=0; tensor.channel> k; k++)
    {
        threads[k]->join();
    }
    return 0;
}