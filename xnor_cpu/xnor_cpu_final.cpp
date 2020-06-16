#include <iostream>
#include <vector>
#include <stdlib.h>
#include <assert.h>  
#include <math.h> 
#include <chrono>
#include <omp.h>
#include <unordered_map>
#include <algorithm> 
#include <functional>

// Indexs are defined with x,y coordinate pairs 
constexpr std::pair<int, int> register_size(8, 8);




// Custom Matrix class using standard vector.
// #TODO Add padding
// #TODO add type conversion function for matrix and tensor
template <typename T>
class Matrix {
    
    typedef std::vector<T> Row;
    
    
public:
	std::vector<Row> data;

    int col, row;
    Matrix(int c, int r): row(r), col(c), data(c, std::vector<T>(r, 0)){}
	Matrix(int c, int r, std::vector<Row> data): row(r), col(c), data(data){}


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
	size_t size()
	{
		return this->data.size();
	}
};




// Custom 3D tensor class using custom Matrix class.
// Scalar multiplication and element-vise multiplication using matrix class
// Hence using Openmp for multi threading with independent matrix objects
template <typename T>
class Tensor{
    typedef Matrix<T> Mat;
	

public:
	std::vector<Mat> tensor;
    int col, row, channel;
    Tensor(int col, int row, int channel): col(col), row(row), channel(channel), tensor(channel, Matrix<T>(col, row)) {}

	Tensor(int col, int row, int channel, std::vector<Mat> tensor): col(col), row(row), channel(channel), tensor(tensor){}


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

	Tensor &operator*=(std::vector<T> row)
    {
        int k;
        #pragma omp parallel private(k) shared(row)
    {
        #pragma omp for schedule(dynamic,50) 
        for (int k=0; channel>k; k++)
        {
            this->tensor[k] = this->tensor[k] * row[k];
        }
    }
    }

	Tensor &operator*=(Matrix<T> mat)
    {
        int k;
        #pragma omp parallel private(k) shared(mat)
    {
        #pragma omp for schedule(dynamic,50) 
        for (int k=0; channel>k; k++)
        {
            this->tensor[k] = this->tensor[k] * mat;
        }
    }
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

	size_t size()
	{
		return this->tensor.size();
	}
};

template<typename T> 
class Weight{
	typedef Tensor<T> tensor;
	
public:
	std::vector<tensor> weight;
	int col, row, channel_in, channel_out;

    Weight(int col, int row, int channel_in, int channel_out): col(col), row(row), channel_in(channel_in), 
	channel_out(channel_out), weight(channel_out, Tensor<T>(col, row, channel_in)) {}

	Weight(int col, int row, int channel_in, int channel_out, std::vector<tensor> weight): col(col), row(row), channel_in(channel_in), 
	channel_out(channel_out), weight( weight) {}

    tensor & operator[](int c)
    {
        return weight[c];
    }
	size_t size()
	{
		return this->weight.size();
	}

};

void countSetBits(int x, int &y) 
{ 

    while (x) { 
        y += x & 1; 
        x >>= 1; 
    } 
} 



void recursive_hash_map(std::vector<int> &key_vector, std::unordered_map<unsigned long int, int> &hash_map,
 int &iteration_count, int count_index, const std::pair<int, int>kernel_size)
{
	for (int i=0; i < iteration_count; i++)
	{
		key_vector[count_index] = i;
		if (count_index == key_vector.size() - 1)
		{
			int key_value = 0;
			int bit_count = 0;
			for(int j=0; j<key_vector.size(); j++)
			{
				key_value += key_vector[j] * std::pow(2, register_size.first * j);  // Shift values so that they can be equal to 
				
			}
			countSetBits(key_value, bit_count);
			hash_map[key_value] = 2* bit_count - ( kernel_size.first * kernel_size.second );
		} 
		else
			recursive_hash_map(key_vector, hash_map, iteration_count, (count_index + 1), kernel_size);
	}
}




std::unordered_map<unsigned long int, int> generate_hash_map(std::pair<int, int> &kernel_size)
{
	std::unordered_map<unsigned long int, int> hash_map;
	std::vector<int> key_vector(kernel_size.second);
	int iteration_count = std::pow(2, kernel_size.first);
	int count_index = 0;
	recursive_hash_map(key_vector, hash_map, iteration_count, count_index, kernel_size);
	return hash_map;
}

template<typename T>
Matrix<T> BinaryMatMemoryAllocation( std::pair<int, int> input_size, std::pair<int, int> kernel_size)
{
	int size_x = ceil((input_size.first - register_size.first) 
						/static_cast<double>(register_size.first + 1 - kernel_size.first) + 1);
	int size_y = ceil((input_size.second - register_size.second ) 
						/static_cast<double>(register_size.second + 1 - kernel_size.second) + 1);
	if (size_x < 0)
		size_x = 1;
	if (size_y < 0)
		size_y = 1;
 
	Matrix<T> binary_mat(size_y, size_x);
	return binary_mat;
}


template<typename T>
void int2binary(const std::vector<T> input_x, const std::pair<int, int> input_index, 
 std::pair<int, int> output_location, unsigned long int &output_y, const std::pair<int ,int>register_size)
{	
	int sign = 0;
	long int pozitive = 1;
	long int negative = 0;
	int count = output_location.second * register_size.second  + output_location.first;

	assert(count < register_size.second * register_size.first); 

	for(int i=0; i<register_size.first; i++)
	{
		sign = (input_x[input_index.first + i] > 0) - (input_x[input_index.first + i] < 0);
		if (sign == 1)
		{
			output_y = pozitive<<count | output_y;
		}
		else if (sign == -1)
		{
			output_y = negative<<count | output_y;
		}
		else
		{
			output_y = negative<<count |output_y;
		}
		if ((input_index.first + i) >=  input_x.size())
		{
			break;
		}
		count++;
		
	}

}


template<typename T>
void intMat2BinaryMat(Matrix<T> &input_mat, Matrix<unsigned long int> &binary_mat, std::pair<int, int> &kernel_size)
{

	int index_x = 0;
	int index_y = 0;
	std::pair<int, int> input_index(0, 0);
	std::pair<int, int> output_location(0, 0);
	
	// Test 
	while(input_mat.size() >= input_index.second)
	{
		std::vector<T> input_row = input_mat[input_index.second];
		int i = 0;
		input_index.first = 0;
		index_x = 0;
		
		while(input_row.size() > i)
		{
			i = input_index.first + register_size.first; 
			int2binary<T>(input_row, input_index, output_location, binary_mat[index_y][index_x], register_size);
			input_index.first = input_index.first + register_size.first + 1 - kernel_size.first;
			index_x++;
			
		}
		output_location.second++;
		input_index.second++;
		if(input_index.second >= input_mat.size())
			{
				break;
			}
		if (output_location.second % register_size.second == 0)
		{
			output_location.second = 0;
			input_index.second = input_index.second + 1 - kernel_size.second;
			index_y++;
		}
	}
}
template<typename T>
Matrix<T> tensorChannelSum(Tensor<T> &input_tensor)
{
	Matrix<T> output_mat(input_tensor.col, input_tensor.row);
	for(int k=0; input_tensor.row > k; k++)
	{
		for(int j=0; input_tensor.col>j; j++)
		{
			int sum = 0;
			for (int i=0; input_tensor.channel>i; i++)
			{
				sum += input_tensor[i][k][j];
			}
			output_mat[k][j] = static_cast<T>(sum) / input_tensor.channel;
		}
	}
	return output_mat;
}

template<typename T>
void tensorChannelSum(Tensor<T> &input_tensor, Matrix<T> &output_matrix)
{
	for(int k=0; input_tensor.row > k; k++)
	{
		for(int j=0; input_tensor.col>j; j++)
		{
			int sum = 0;
			for (int i=0; input_tensor.channel>i; i++)
			{
				sum += input_tensor[i][k][j];
			}
			output_matrix[k][j] = static_cast<T>(sum) / input_tensor.channel;
		}
	}
}

void cellConv2D(unsigned long input_mat, unsigned long conv_kernel, const unsigned long mask,
				std::pair<int, int> conv_iter, std::pair<int, int> output_index, std::pair<int, int> image_size,
				Matrix<unsigned long> &output_mat)
{
	const std::pair<int, int> input_index(0, 0);
	long int shifter = 0;
	
	int sign = 0;
	int index_x = 0;
	 
	 // iteration parameters for convolution kernel
	 // X axis calculation
	if ( output_index.first + register_size.first < image_size.first)
	{
		conv_iter.first = conv_iter.first;
	}
	else
	{
		conv_iter.first = conv_iter.first - (register_size.first + output_index.first - image_size.first);
	}
	// Y axis calculation
	if 	((output_index.second  + register_size.second) < image_size.second)
	{
		conv_iter.second = conv_iter.second;
	}
	else
	{
		conv_iter.second = conv_iter.second - (register_size.second + output_index.second - image_size.second);
	}
	unsigned long int shift = 0;
	for (int j=0; conv_iter.second > j; j++)
	{
		for (int i=0; conv_iter.first > i; i++)
		{
			// Convolution operation here
			shifter = i + j * register_size.first;
			output_mat[output_index.second + j][output_index.first + i] = (input_mat | (conv_kernel>>shifter))&mask; // This is wrong, input matrix must be shifted.
		}
	}

}



void binaryConv2D(Matrix<unsigned long> input_mat, Matrix<unsigned long> &output_mat,
			unsigned long conv_kernel, std::pair<int, int> conv_size,
			std::pair<int, int> image_size)
{

	unsigned long int mask = 0;
	mask = std::pow(2, conv_size.first) - 1;
	for (int j=1; conv_size.second > j ; j++)
	{
		mask = (mask<<register_size.second) | static_cast<unsigned long>(std::pow(2, conv_size.first) - 1);
	}
	// mask = 1110000011100000111 = 2^3 -1 - 2^8 + 2^11 - 2^ 

	const int conv_per_row = register_size.first - (conv_size.first - 1);
	const int conv_per_column = register_size.second - (conv_size.second - 1);
	std::pair<int, int> conv_iter = std::make_pair(conv_per_row, conv_per_column);
	std::pair<int, int> output_index(0, 0); 
	for(int j=0; input_mat.size()>j; j++)
	{
		output_index.first = 0;
		for(int i=0; input_mat[0].size()>i; i++)
		{
			cellConv2D(input_mat[j][i], conv_kernel, mask,
				 		conv_iter, output_index, image_size,
						output_mat);
			output_index.first += conv_per_row;
		}
		output_index.second += conv_per_column;

	}
}
template<typename T>
void conv2D(Matrix<T> &input_x,  Matrix<T> &output_y, std::pair<int, int>kernel_size)
{
	std::vector<std::vector<float>> scaled_x(input_x.size(), std::vector<float>(input_x[0].size(), 0));

	float k = 1.0 / (input_x.size() * input_x[0].size());
	
	for (int j=0; input_x.size()>j; j++)
	{
		for (int i=0; input_x[0].size()>i; i++)
		{
			scaled_x[j][i] = input_x[j][i] * k;
			for(int y=(1 - kernel_size.second)/2 ; (kernel_size.second + 1)/2 > y; y++)
			{
				if ( ((j+ y)<0) || ((j + y )>= static_cast<int>(output_y.size())) )
					{
						break;
					} 
				for(int x=(1- kernel_size.first)/2 ; (kernel_size.first + 1)/2 > x; x++)
				{

					if (((i+ x)<0) || ((i + x)>=static_cast<int>(output_y[0].size()) ) )
					{
						break;
					}
					output_y[j+y][i+x] += scaled_x[j][i]; 
				}

			}
			
		}
	}
}

void binaryMat2IntMat(Matrix<unsigned long> &input_x, const std::unordered_map<unsigned long, int> &hash_map)
{
	for(int j=0; input_x.size()>j; j++)
	{
		for (int i=0; input_x[j].size()>i; i++)
		{
			input_x[j][i] = hash_map.at(input_x[j][i]);
		}
	}
}
// output matrix must be have zero as defualt value
template<typename T>
void ZeroPadding2D(Matrix<T> &input_mat, Matrix<T> &output_mat, std::pair<int, int> kernel_size)
{
	for(int j=0; input_mat.row>j; j++)
	{
		for(int i=0; input_mat.col>i; i++)
		{
			output_mat[j + (kernel_size.second - 1)/2][i + (kernel_size.first - 1)/2] = input_mat[j][i];
		}
	}
}	
// #TODO# learn std::sharedptr, Rvalue refence and std::move  

// A Xnor Convolution layer is made of:
// 1- int input matrix and weight matrix to binary input matrix
// 2- Calculate K matrix ([1,3], and 2 can be concurrent execution)
// 3- Binary Convolution
// 4- Hash table conversion
// 5- Output of Hash table * K matrix
// 6- scalar * output of [5]
// 7- Result Matrix 

// A xnor convolution function needs input and weight matrix, alpha, hash table(Can calculate inside but performance)
// A xnor convolution outputs result matrix
// xnor_convolution does not include pooling if needed ended padded input image.
// A padding can be added to xnor_convolution to increase performance 
// however it may cause some unstabilities and need testing and more time.


template<typename in, typename out>
void MatTypeCasting(Matrix<in>& input_mat, Matrix<out>& output_mat)
{

	for (int j=0; input_mat.col>j; j++)
	{
		for(int i=0; input_mat.row>i; i++)
		{
			output_mat[j][i] = static_cast<out>(input_mat[j][i]);
		}
	}
}

template<typename T>
void xnor_convoltion_op(Matrix<T> input_matrix, Matrix<T> &output_matrix, Matrix<float> &weight_matrix,
 std::unordered_map<unsigned long, int> hash_map, bool padding = true)
{
	auto kernel_size = std::make_pair(weight_matrix.col, weight_matrix.row);
	if (padding == true)
	{
		Matrix<T> padded_matrix(input_matrix.row + kernel_size.first -1, input_matrix.col + kernel_size.second -1);
		ZeroPadding2D(input_matrix, padded_matrix, kernel_size);
		input_matrix = padded_matrix; 
	}
	
	
	auto binary_input_matrix = BinaryMatMemoryAllocation<unsigned long>(std::make_pair(input_matrix.row, input_matrix.col), kernel_size);
	auto binary_weight_matrix = BinaryMatMemoryAllocation<unsigned long>(std::make_pair(weight_matrix.row, weight_matrix.col), kernel_size);
	intMat2BinaryMat<T>(input_matrix, binary_input_matrix, kernel_size);
	intMat2BinaryMat<T>(weight_matrix, binary_weight_matrix, kernel_size);
	Matrix<unsigned long int> output_mat(input_matrix.col - weight_matrix.col + 1, input_matrix.row - weight_matrix.row + 1);
	binaryConv2D(binary_input_matrix, output_mat, binary_weight_matrix[0][0],
	 kernel_size, std::make_pair(input_matrix.col, input_matrix.row));
	binaryMat2IntMat(output_mat, hash_map);
	MatTypeCasting<unsigned long, T>(output_mat, output_matrix);
}

template<typename T>
Tensor<T> xnor_convoltion(Tensor<T> &input_tensor, Weight<T> &weight, std::vector<float> &alpha,
std::unordered_map<unsigned long, int> hash_map, bool padding = true)
{
	int output_row;
	int output_col;
	int output_channel;

	if (padding == true)
	{
		output_row = input_tensor.row;
		output_col = input_tensor.col;
		output_channel = weight.channel_out;
	}

	else
	{
		output_row = input_tensor.row + 1 - weight.row;
		output_col = input_tensor.col + 1 - weight.col;
		output_channel = weight.channel_out;
	}
	Tensor<T> output_tensor(output_row, output_col, output_channel);

	auto A = tensorChannelSum<float>(input_tensor);
	Matrix<T> padded_matrix(output_tensor.row + weight.row -1, output_tensor.col + weight.col -1);
	ZeroPadding2D(A, padded_matrix, std::make_pair(weight.row, weight.col));
	Matrix<float> K(padded_matrix.col - weight.col + 1, padded_matrix.row - weight.row + 1);
	conv2D<float>(padded_matrix, K, std::make_pair(weight.row, weight.col));
	{
		int in, out;
		Weight<T> output_tensor_buffer(output_tensor.row, output_tensor.col, weight.channel_in, weight.channel_out);
		auto start = std::chrono::high_resolution_clock::now();
		#pragma omp parallel private(in, out) shared(input_tensor, output_tensor_buffer, hash_map)
 		{
			#pragma omp for schedule(dynamic,50) collapse(2)
			for(int out=0; weight.channel_out>out; out++)
			{
				
				for (int in=0; weight.channel_in>in; in++)
				{
					xnor_convoltion_op<float>(input_tensor[in], output_tensor_buffer[out][in], weight[out][in], hash_map);

				}
				// auto A = tensorChannelSum<float>(input_tensor);
				// output_tensor[out] *= K;
				// output_tensor[out] *= alpha[out];
			}
		}
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> multi_core(stop - start);
		std::cout<<"Multicore xnor Kernel Performance : "<<multi_core.count()<<std::endl;
		#pragma omp parallel private(out) shared(output_tensor_buffer, output_tensor, K, alpha)
		{
		#pragma omp for schedule(dynamic,50) collapse(1)
		for (out=0; output_tensor_buffer.channel_out>out; out++)
		{
			tensorChannelSum<float>(output_tensor_buffer[out], output_tensor[out]);
			output_tensor[out] *= K;
			output_tensor[out] *= alpha[out];

		}
		}
	}
	return output_tensor;
}
int main()
{
	Tensor<float> input_tensor(512, 512, 64);
	Weight<float> weight(3, 3, input_tensor.channel, 64);
	std::vector<float> scalar(weight.channel_out);
	std::pair<int, int> kernel_size(weight.row, weight.col);
	// Random initilizate the values
	for(int k=0; input_tensor.channel>k; k++)
	{
		for(int j=0; input_tensor.col>j; j++)
		{
			for(int i=0; input_tensor.row>i; i++)
			{
				input_tensor[k][j][i] = (std::rand()%1000 - 500);
				if (input_tensor[k][j][i] >= 0)
				{
					input_tensor[k][j][i] = 1;
				}
				else
				{
					input_tensor[k][j][i] = -1;
				}
				
			}
		}
	}
	for (int m=0; weight.channel_out>m; m++)
	{
		scalar[m] = static_cast<float>(rand()) / static_cast<float> (RAND_MAX);
		for(int k=0; weight.channel_in>k; k++)
		{
			
			for(int j=0; weight.col>j; j++)
			{
				for(int i=0; weight.row>i; i++)
				{
					weight[m][k][j][i] = (std::rand()%1000 - 500);
					if (weight[m][k][j][i] >= 0)
					{
						weight[m][k][j][i] = 1;
					}
					else
					{
						weight[m][k][j][i] = -1;
					}
					
				}
			}
		}
	}
	// Calculate hash map
	auto hash_map = generate_hash_map(kernel_size);
	auto start = std::chrono::high_resolution_clock::now();
	auto output_tensor = xnor_convoltion<float>(input_tensor, weight, scalar, hash_map);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> multi_core(stop - start);
	std::cout<<"Multi core Xnor Performance : "<<multi_core.count()<<std::endl;
	return 0;
}

