 #include "vector"

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
    std::pair<int,int> in_size = get_matrix_shape<T>(in);
    //std::cout << in.size() << std::endl;
    //std::cout << in[0].size() << std::endl;
    for (int i = 0; i < in_size.first; i++)
    {
        // in channel
        for (int j = 0; j < in_size.second; j++)
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


    for (int i = 0; i < input_size.first; i++)
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

    return output_matrix;
    
}


template<typename T>
void conv2D(matrix_3d<T> input_matrix, matrix_4d<T> weight_matrix ){
    
    std::pair<int,int> channel_dims = get_matrix_shape(weight_matrix);
    std::pair<int,int> kernel_dims = get_matrix_shape(weight_matrix[0][0]);
    matrix_3d<T> output_matrix;
    
    // out channel 
    for (int i = 0; i < channel_dims.first; i++)
    {
        // in channel
        matrix_2d<T> out(kernel_dims.first, matrix_1d<T>(kernel_dims.second,0));
        zero_initialize_2D(out, kernel_dims);
        for (int j = 0; j < channel_dims.second; j++)
        {
            sum<T> (out,conv_op<T>(input_matrix[j],weight_matrix[i][j]));
        }
        
        output_matrix.push_back(out);
        out.clear();
    }
    int x =1;
    
}