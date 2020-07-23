#include "vector"
#include "chrono"
#include "iostream"
#include <omp.h>


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

template<typename T>
std::pair<int, int> get_matrix_shape(matrix_2d<T> matrix){
    
    int height = matrix.size();
    int width = matrix[0].size();
    return std::make_pair(height,width);
}

static double total_time = 0; 

template<typename T>
void zero_padding_2D(matrix_2d<T> &input_mat, matrix_2d<T> &output_mat, std::pair<int, int> input_size, std::pair<int, int> kernel_size)
{

	for(int i=0; i < input_size.first; i++)
	{
		for(int j=0; j < input_size.second; j++)
		{
			output_mat[i + (kernel_size.first - 1)/2][j + (kernel_size.second - 1)/2] = input_mat[i][j];
		}
	}
}


template<typename T>
void zero_initialize_2D(matrix_2d<T> &output_mat, std::pair<int, int> output_size)
{
    matrix_1d<T> dummy;
    for(int i=0; i < output_size.first; i++)
	{
        
		for(int j=0; j < output_size.second; j++)
		{
            output_mat[i][j] = 0;
		}

	}
}


template<typename T>
void sum(matrix_2d<T> &out, matrix_2d<T> in)
{
    std::pair<int,int> out_size = get_matrix_shape<T>(out);
    //std::cout << in.size() << std::endl;
    //std::cout << in[0].size() << std::endl;
    for (int i = 0; i < out_size.first; i++)
    {
        // in channel
        for (int j = 0; j < out_size.second; j++)
        {
            out[i][j] += in[i][j];
        }
    }
    
}


template<typename T>
matrix_2d<T> conv_op(matrix_2d<T> input_matrix, matrix_2d<T> kernel_matrix)
{

    std::pair<int, int> kernel_size = get_matrix_shape<T>(kernel_matrix);
    std::pair<int, int> input_size = get_matrix_shape<T>(input_matrix);
    
    matrix_2d<T> output_matrix(input_size.first, matrix_1d<T> (input_size.second,0));
    zero_initialize_2D(output_matrix,input_size);

    //matrix_2d<T> padded_matrix(input_size.first + kernel_size.first -1, input_size.second + kernel_size.second -1);
    
    matrix_2d<T> padded_matrix(input_size.first + kernel_size.first -1, matrix_1d<T>(input_size.second + kernel_size.second -1,0));
    zero_padding_2D(input_matrix, padded_matrix, input_size, kernel_size);
    std::pair<int,int> padded_size = get_matrix_shape<T>(padded_matrix);


    int i; 
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel private(i) shared(padded_matrix, kernel_matrix, output_matrix)
    {
    #pragma omp for schedule(dynamic,50) collapse(1)

    for (i = 0; i < input_size.first; i++)
    {
        for (int j = 0; j < input_size.second; j++)
        {
            for (int k = 0; k < kernel_size.first; k++)
            {
                for (int z = 0; z < kernel_size.second; z++)
                {
                    output_matrix[i][j] += padded_matrix[i+k][j+z] * kernel_matrix[k][z];
                }
            }
        }
        
    }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> multi_core(stop - start);
    //std::cout<<"Time spend for Convolution :"<< multi_core.count()<< std::endl;
    total_time += static_cast<double>(multi_core.count());
    return output_matrix;
    
}


template<typename T>
matrix_3d<T> conv2D(matrix_3d<T> input_matrix, matrix_4d<T> weight_matrix,int row, int col){
    
    std::pair<int,int> channel_dims = get_matrix_shape(weight_matrix);
    std::pair<int,int> kernel_dims = get_matrix_shape(weight_matrix[0][0]);
    matrix_3d<T> output_matrix;
    
    // out channel 
    for (int i = 0; i < channel_dims.first; i++)
    {
        // in channel
        
        matrix_2d<T> out(row, matrix_1d<T>(col,0));
        zero_initialize_2D(out, kernel_dims);
        for (int j = 0; j < channel_dims.second; j++)
        {
            sum<T> (out,conv_op<T>(input_matrix[j],weight_matrix[i][j]));
        }
        
        output_matrix.push_back(out);
        out.clear();
    }
    return output_matrix;
    
}


int main()
{
    int row[] = {128, 256, 512, 1024, 2048};
    int col[] = {128, 256, 512, 1024, 2048};
    int kernel = 3;
    int channel_in = 1;
    int channel_out = 1;
    int total_test_count = 100;
    for(int i=0; i<5; ++i)
    {
        for (int j=0; j<total_test_count; ++j)
        {
        //std::cout << "Row size"<< row[i] << std::endl;
        matrix_3d<double> input_matrix(channel_in, matrix_2d<double>(row[i], std::vector<double>(col[i], 0) ) );
        matrix_4d<double> weight_matrix(channel_out, matrix_3d<double>(channel_in, std::vector<std::vector<double>>(kernel, std::vector<double>(kernel, 0) ) ) ) ;
        auto output = conv2D<double>(input_matrix, weight_matrix, row[i], col[i]);
        }
        total_time = total_time / static_cast<double>(total_test_count);
        std::cout<< "Averaged Time "<<total_time << "  For Row size:"<< row[i] << std::endl;
        total_time = 0;
    }
}