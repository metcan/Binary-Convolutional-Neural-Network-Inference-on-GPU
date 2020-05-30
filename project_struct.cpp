#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <omp.h>


template <typename T>
class Matrix {
    
    typedef std::vector<T> Row;
    
    std::vector<Row> data;

public:
    int col, row;
    Matrix(int c, int r): row(r), col(c), data(c, std::vector<T>(r)){}

    Matrix(int c, int r, std::vector<Row> & data): row(r), col(c), data(data){}


    Row & operator[](int i)
    {
        return data[i];
    }
    Matrix operator*(T scalar)
    {
        Matrix<T> mat(col, row);
        for (int j=0; col>j; j++)
        {
            for (int i=0; row>i; i++)
            {
                mat[j][i] = this->data[j][i] * scalar; 

            }
        }
        return mat;

    }
    Matrix &operator*=(T scalar)
    {
        for (int j=0; col>j; j++)
        {
            for (int i=0; row>i; i++)
            {
                this->data[j][i] = this->data[j][i] * scalar; 

            }
        }
    }
    Matrix operator*(Matrix<T> matrix)
    {
        Matrix<T> mat(col, row);
        for (int j=0; col>j; j++)
        {
            for (int i=0; row>i; i++)
            {
                mat[j][i] = this->data[j][i] * matrix[j][i]; 

            }
        }
        return mat;

    }
    Matrix &operator*=(Matrix<T> matrix)
    {
        for (int j=0; col>j; j++)
        {
            for (int i=0; row>i; i++)
            {
                this->data[j][i] = this->data[j][i] * matrix[j][i]; 

            }
        }
    }
};

template <typename T>
class Tensor{
    typedef Matrix<T> Mat;

    std::vector<Mat> tensor;

public:
    int col, row, channel;
    Tensor(int col, int row, int channel): col(col), row(row), channel(channel), tensor(channel, Matrix<T>(col, row)) {}


    Mat & operator[](int c)
    {
        return tensor[c];
    }
    Tensor operator*(T scalar)
    {
        Tensor<T> tensor(col, row, channel);
        int k;
        #pragma omp parallel private(k) shared(scalar, tensor) 
        {
        #pragma omp for schedule(dynamic, 50)
        for (k=0; channel>k; k++)
        {
            tensor[k] = this->tensor[k] * scalar;
        }
        }
        return tensor;



    }
    Tensor &operator*=(T scalar)
    {
        int k;
        #pragma omp parallel private(k) shared(scalar)
    {
        #pragma omp for schedule(dynamic,50) 
        for (int k=0; channel>k; k++)
        {
            this->tensor[k] = this->tensor[k] * scalar;
        }
    }
    }
    Tensor operator*(Tensor<T> tensor)
    {
        Tensor<T> ten(col, row, channel);
        int k;
        #pragma omp parallel private(k) shared(tensor)
    {
        #pragma omp for schedule(dynamic,50) 
        for (int k=0; channel>k; k++)
        {
            ten[k] = this->tensor[k] * tensor[k];
        }

    }
            return ten;
    }
    Tensor &operator*=(Tensor<T> tensor)
    {

        int k;
        #pragma omp parallel private(k) shared(tensor)
    {
        #pragma omp for schedule(dynamic,50) 
        for (int k=0; channel>k; k++)
        {
            this->tensor[k] = this->tensor[k] * tensor[k];
        }

    }
    }


};

template<typename T> 
class conv2DWeights{
	typedef Tensor<T> tensor;
	std::vector<tensor> weight;
public:
	int col, row, channel_in, channel_out;
    conv2DWeights(int col, int row, int channel_in, int channel_out): col(col), row(row), channel_in(channel_in), 
	channel_out(channel_out), weight(channel_out, Tensor<T>(col, row, channel_in)) {}
    tensor & operator[](int c)
    {
        return weight[c];
    }

};

template <typename T>
void TensorSquare(Tensor<T> &tensor)
{
    int k = 0, j = 0, i = 0;
    //#pragma omp parallel private(i, j, k) shared(tensor)
    //#pragma omp for schedule(dynamic,50) 
    for(k=0; tensor.channel> k; k++)
    {
        for(j=0; tensor.col> j; j++)
        {
            for(i=0; tensor.row > i; i++)
            {
                tensor[k][j][i] = tensor[k][j][i] * tensor[k][j][i];
            }
        }
    }
}


template <typename T>
void MatrixSquare(Matrix<T> &mat)
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
    Tensor<int> tensor(512, 512, 512);
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
    std::cout << "start from single" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    TensorSquare<int>(tensor); 
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> single_core(stop - start);
    std::cout << single_core.count() << std::endl;
    // MULTI THREAD PERFORMANCE
    auto start_1 = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel
    int k = 0;
    #pragma omp parallel private(k) shared(tensor)
    {
        #pragma omp for schedule(dynamic,50) 
        for (k=0; tensor.channel > k ; k++)
            {
                tensor[k] *= tensor[k];
            }
    }
    tensor *= 5;
    tensor *= tensor;
    auto ten = tensor * tensor;
    ten = tensor * 5;
    auto stop_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> multi_core(stop_1 - start_1);
    std::cout << multi_core.count() << std::endl;

    

    return 0;
}