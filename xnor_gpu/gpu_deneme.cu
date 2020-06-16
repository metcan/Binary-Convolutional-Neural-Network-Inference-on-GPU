#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <vector>
#include <assert.h>
#include <math.h>

constexpr std::pair<int, int> register_size(8, 4);


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



std::pair<int, int> to_binary_size(std::pair<int, int>input_size,  std::pair<int, int>kernel_size){
	int size_x = ceil((input_size.first - register_size.first)
						/static_cast<double>(register_size.first + 1 - kernel_size.first) + 1);
	int size_y = ceil((input_size.second - register_size.second )
						/static_cast<double>(register_size.second + 1 - kernel_size.second) + 1);
	if (size_x < 0)
		size_x = 1;
	if (size_y < 0)
		size_y = 1;
	return std::make_pair(size_x, size_y);
}
template<typename T>
class Matrix1d{
public:
	int lenght;
	T* data;
	Matrix1d()
    {
        lenght = 64;
        data = nullptr;
    }
	Matrix1d(int lenght, T *data): lenght(lenght), data(data){}
	~Matrix1d()
	{
		delete[] data;
		delete data;
	}
};

template<typename T>
class Matrix2d{
public:
	int row;
	int col;
	T** data;
	Matrix2d()
    {
        row = 64;
        col = 64;
        data = nullptr;
    }
	Matrix2d(int row, int col, T** data): row(row), col(col), data(data){}
	~Matrix2d()
	{
		for (int i=0; row>i; ++i)
		{
			delete[] data[i];
		}
		delete[] data;
		delete data;

	}
};

template<typename T>
class Matrix3d{
public:
	int channel;
	int row;
	int col;
	T*** tensor;
	Matrix3d()
    {
        row = 64;
        col = 64;
        channel = 64;
        tensor = nullptr;
    }
	Matrix3d(int channel, int row, int col, T*** tensor): channel(channel), row(row), col(col), tensor(tensor){}
	~Matrix3d()
	{
		for (int k=0; channel> k; ++k)
		{
			for (int j=0; row>j; ++j)
			{
				delete[] tensor[k][j];
			}
			delete[] tensor[k];
		}
		delete[] tensor;
		delete tensor;
	}
};

//don't modify
const int nTPB=256;

template <typename T>
__inline__ __device__
T warpReduceSum(T val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset); // requires CUDA 9+
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val) {

  static __shared__ T shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSum(val);     // Each warp performs partial reduction
  if (lane==0) shared[wid]=val; // Write reduced value to shared memory
  __syncthreads();              // Wait for all partial reductions

                      //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
  return val;
}

size_t choose_block_size(size_t val){
  if (val >= nTPB) return nTPB;
  if (val <= 32) return 32;
  val = (val >> 1) | val;
  val = (val >> 2) | val;
  val = (val >> 4) | val;
  val = (val >> 8) | val;
  val = (val >> 16) | val;
  val++;
  return val;
}

template<typename T>
void __global__ kernel_reduce_sum(
		const T * __restrict__  d_idata,
		float * __restrict__ d_odata,
        const int dim0,
        const int dim1,
        const int dim2)
{
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < (dim0*dim1)){
	  int tidx = idx;
	  float tsum = 0;
	  for (int i = 0; i < dim2; i++)
	  {
		tsum += static_cast<float>(d_idata[tidx]);
		tidx += dim0*dim1;
	  }
	  d_odata[idx] = tsum / static_cast<float>(dim2);
	}
}

template<typename T>
 __device__ void to_binary_register(
	const T &idata,
	unsigned int &odata,
	 int *output_location)
{
	int sign = (idata > 0) - (idata < 0);
	const unsigned int pozitive = 1;
	const unsigned int negative = 0;
	//int count = output_location[1] * register_size.second  + output_location[0];
	//assert(count < register_size.second * register_size.first);
	if (sign > -1)
	{
		odata = pozitive<<(output_location[1] * register_size.first  + output_location[0]) | odata;
	}
	else
	{
		odata = negative<<(output_location[1] * register_size.first  + output_location[0]) | odata;
	}
}

template<typename T>
void __global__  to_binary_tensor(
	const T *  d_idata,
	unsigned int *  d_odata,
	const int row, const int b_row,
	const int col, const int b_col,
	const int kernel_row = 3, const int kernel_col = 3)
{
	// Each thread will store a size = 32 array inside their single register
	int idx = threadIdx.x+blockDim.x*blockIdx.x; //register IDX
	// n*(regsiter_size - kernel_size)
	if (idx < (b_row * b_col))
	{
		int input_index[] = {(idx%b_col) * (register_size.first - kernel_col), (idx /b_col ) * (register_size.second - kernel_row)};
		int data_idx = input_index[0] + (input_index[1] * row);
		//int input_index[] = {data_idx%row, data_idx/col, data_idx/(row*col)}; // from start of array , (x, y, z)
		int register_location[] = {0, 0};
		unsigned int local_register;
		for (int j=0; register_size.second>j; j++)
		{
			for (int i=0; register_size.first>i; i++)
			{
				to_binary_register<T>(d_idata[data_idx], local_register, register_location);
				++data_idx;
				input_index[0] += 1;
				register_location[0] += 1;
				if (input_index[0] == col) break;
			}
			data_idx = data_idx + col - register_location[0];
			input_index[1] += 1;
			input_index[0] = (idx%b_col) * (register_size.first - kernel_col);
			register_location[0] = 0;
			register_location[1] += 1;
			if (input_index[1] == row) break;
		}
		d_odata[idx] = local_register;
	}
}

// For each Register launch a kernel
void __global__ binaryConv2d(
		const unsigned int * binary_mat,
		int * output_mat,
		const unsigned int *weight_matrix,
		int binary_row, int binary_col,
		int kernel_row, int kernel_col,
		int output_row, int output_col
		)
{

	int idx = threadIdx.x +blockDim.x*blockIdx.x; //binary Cell id
	int conv_per_row = register_size.second - (kernel_row - 1);
	int conv_per_column = register_size.first - (kernel_col - 1);
	int output_index_x = (idx % binary_col) * conv_per_column;
	int output_index_y = (idx / binary_col) * conv_per_row;

	if ( (output_index_x + conv_per_column) > output_col)
	{
		conv_per_column = output_col - output_index_x;
		if (conv_per_column < 0)
		{
			int x = 5;
		}
	}
	if ( (output_index_y + conv_per_row) > output_row)
	{
		conv_per_row = output_row - output_index_y;
	}

	unsigned int mask = std::pow(2, kernel_col) - 1;

	for (int j=1; kernel_row > j; j++)
	{
		mask = (mask<<register_size.first) | static_cast<unsigned int>(std::pow(2, kernel_col) - 1);
	}


	unsigned int shifter = 0;
	for (int j=0; conv_per_row>j; ++j)
	{
		for (int i=0; conv_per_column>i; ++i)
		{

			output_mat[(output_index_y+j)*output_col + output_index_x + i] = (~(binary_mat[idx]>>shifter) ^ (weight_matrix[0]) ) & mask;
			++shifter;
		}
		// Check if register is not fully filled,
		// if not add shifter the missing shift amount
		shifter += register_size.second - conv_per_column;
	}

}

template<typename T>
void __global__ zeroPadding(T * input_mat, T * output_mat,  int kernel_row, int kernel_col, int matrix_row, int matrix_col)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int index_x = (idx % matrix_col) - (kernel_row - 1)/ 2;
	int index_y = (idx/matrix_row) - (kernel_col - 1)/ 2;
	if(index_x > 0 || index_y>0 )
	{
		if( index_x< matrix_row || index_y < matrix_row)
		{
			output_mat[idx] = input_mat[index_y * matrix_col + index_x];
		}
	}
	else output_mat[idx] = 0;
}
void __global__ binary2int(const  int *input_mat, float * output_mat,  int matrix_row, int matrix_col, int kernel_row, int kernel_col)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < matrix_row* matrix_col)
	{
		unsigned int mask = 1;
		unsigned int shifter = 0;
		int buffer = 0;
		int __restrict__ data = input_mat[idx];
		for (int j=0; kernel_row>j; ++j)
		{
			for(int i=0; kernel_col>i; ++i)
			{
				buffer += (data >> shifter) & mask;
				++shifter;
			}
			shifter += register_size.first - kernel_col;
		}
		output_mat[idx] = static_cast<float>(2 * buffer - (kernel_row * kernel_col) ) ;
	}
}

int main()
{

	// Initialize a 3d tensor
	int row = 4;
	int col = 8;
	int channel_in = 1;
	int kernel_row = 3;
	int kernel_col = 3;
	int channel_out = 1;
	float *h_conv_output = new float[row * col];
	float *d_conv_output;
	cudaMalloc(&d_conv_output, row * col * sizeof(float));


	// Test Channel Summation
	int *h_tensor = new int[channel_in * row * col];
	float *h_matrix = new float[row * col];
	int	*h_weights = new int[kernel_row * kernel_col * channel_in * channel_out];
	for (int i=0; channel_in* row * col>i; ++i) h_tensor[i] = (rand() % 50) - 10;
	for (int i=0; channel_in * channel_out * kernel_row * kernel_col>i; ++i) h_weights[i] = (rand() % 50) - 10 ;
	//for (int i=0; kernel_row * kernel_col * channel_in * channel_out > i; ++i) (rand() % 100) - 50;

	// Channel summation
	assert( h_tensor != NULL);
	size_t sz = row * col * channel_in* sizeof(int);
	size_t rsz = row * col * sizeof(int);
	int *d_tensor;
	float *d_matrix;
	cudaError_t err = cudaMalloc(&d_tensor, sz);
	if (err != cudaSuccess) {printf("cudaMalloc1 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMalloc(&d_matrix, rsz);
	if (err != cudaSuccess) {printf("cudaMalloc2 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMemcpy(d_tensor, h_tensor, sz, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {printf("cudaMemset1 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMemset(d_matrix, 0 , rsz);
	if (err != cudaSuccess) {printf("cudaMemset2 error: %s\n", cudaGetErrorString(err)); return -1;}
	auto block_size = choose_block_size(row * col);
	kernel_reduce_sum<int> <<<((row * col)+(block_size-1))/block_size, block_size>>> (d_tensor, d_matrix, col, row, channel_in);
	err = cudaMemcpy( h_matrix, d_matrix, rsz, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {printf("result1 error: %s\n", cudaGetErrorString(err)); return -1;}
	std::cout << "Channel Sum Done" <<std::endl;
	//int * h_paddedTensor = new int[(row+ kernel_row - 1)* (col + kernel_col - 1)];
	int * d_paddedTensor;
	auto padded_size = sizeof(int) * (row+ kernel_row - 1)* (col + kernel_col - 1);
	cudaMalloc(&d_paddedTensor, padded_size);
	block_size = choose_block_size(row+ kernel_row - 1 * col + kernel_col - 1);

	zeroPadding<int><<<((row+ kernel_row - 1 + col + kernel_col - 1)+(block_size-1))/block_size, block_size>>> (d_tensor, d_paddedTensor, kernel_row, kernel_col, row+ kernel_row - 1, col + kernel_col - 1);

	cudaFree(d_tensor);
	d_tensor = d_paddedTensor;
	delete h_tensor;
	cudaMemcpy(h_tensor, d_tensor, rsz, cudaMemcpyDeviceToHost);
	row = row+ kernel_row - 1;
	col = col + kernel_col - 1;


	std::cout << "Zero Padding" << std::endl;
	// float|int to binary
	auto binary_size = to_binary_size(std::make_pair(col, row), std::make_pair(kernel_col, kernel_row));
	unsigned int *h_binary_tensor = new unsigned int[channel_in* binary_size.second * binary_size.first];
	unsigned int *d_binary_tensor;
	size_t bsz =  binary_size.first * binary_size.second * channel_in* sizeof(unsigned int);
	cudaMalloc(&d_binary_tensor, bsz);
	block_size = choose_block_size(binary_size.first * binary_size.second * channel_in);
	to_binary_tensor<int><<<(binary_size.first * binary_size.second * channel_in+ block_size - 1)/block_size, block_size>>>(d_tensor, d_binary_tensor, row, binary_size.second, col, binary_size.first, kernel_row, kernel_col);
	cudaMemcpy(h_binary_tensor, d_binary_tensor, bsz, cudaMemcpyDeviceToHost);

	std::cout << "Input Tensor to Binary Done" << std::endl;
	int *d_weights;
	auto weight_size = sizeof(int) * channel_in *channel_out * kernel_col * kernel_row;
	cudaMalloc(&d_weights, weight_size);
	binary_size = to_binary_size(std::make_pair(kernel_col, kernel_row), std::make_pair(kernel_col, kernel_row));
	cudaMemcpy(d_weights, h_weights, weight_size, cudaMemcpyHostToDevice);
	unsigned int *h_binary_weight = new unsigned int[channel_out * channel_in * binary_size.second * binary_size.first];
	unsigned *d_binary_weight;
	bsz =  binary_size.first * binary_size.second * channel_in* sizeof(unsigned int);
	cudaMalloc(&d_binary_weight, bsz);
	block_size = choose_block_size(binary_size.first * binary_size.second * channel_in * channel_out);
	to_binary_tensor<int><<<(binary_size.first * binary_size.second * channel_in * channel_out + block_size - 1)/block_size, block_size>>>(d_weights, d_binary_weight,
						kernel_row, binary_size.second, kernel_col, binary_size.first, kernel_row, kernel_col);
	cudaMemcpy(h_binary_weight, d_binary_weight, bsz, cudaMemcpyDeviceToHost);

	std::cout << "Input Weight to Binary Weight Done" << std::endl;
	// Binary convolution
	int binary_row = row - kernel_row + 1;
	int binary_col = col - kernel_col + 1;
	int * h_binary_output = new int[ binary_row * binary_col ];
	int * d_binary_output;
	auto output_mat_size = binary_col * binary_row * sizeof(int);
	cudaMalloc(&d_binary_output, output_mat_size);
	block_size = choose_block_size(binary_size.first * binary_size.second);
	binary_size = to_binary_size(std::make_pair(col, row), std::make_pair(kernel_col, kernel_row));
	binaryConv2d<<<(binary_size.first * binary_size.second + block_size - 1)/ block_size ,block_size>>>(d_binary_tensor, d_binary_output, d_binary_weight,
																										binary_size.second, binary_size.first,
																										kernel_row, kernel_col,
																										binary_row, binary_col);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(h_binary_output, d_binary_output, output_mat_size, cudaMemcpyDeviceToHost));
	std::cout << "Xnor convolution  Done" << std::endl;

	for (int j=0; binary_size.second >j; ++j)
	{
		for (int i= 0; binary_size.first>i; ++i)
		{
			std::cout<< h_binary_tensor[j* binary_size.first + i]<< " ";
		}
		std::cout<< "\n";
	}
	std::cout<< *h_binary_weight << std::endl;
	block_size = choose_block_size(binary_row * binary_col);
	binary2int<<<(binary_row * binary_col + block_size - 1)/ block_size ,block_size>>>(d_binary_output, d_conv_output, binary_row, binary_col, kernel_row, kernel_col);
	cudaMemcpy(h_conv_output, d_conv_output, binary_row * binary_col* sizeof(float), cudaMemcpyDeviceToHost);


	for (int j=0; binary_row >j; ++j)
	{
		for (int i= 0; binary_col>i; ++i)
		{
			std::cout<< h_conv_output[j* binary_col + i]<< " ";
		}
		std::cout<< "\n";
	}
	std::cout<<std::endl;
	cudaFree(d_binary_output);
	cudaFree(d_binary_tensor);
	cudaFree(d_binary_weight);
	cudaFree(d_conv_output);
	cudaFree(d_paddedTensor);
	cudaFree(d_matrix);
	cudaFree(d_weights);
	cudaFree(d_tensor);
	return 0;
}

