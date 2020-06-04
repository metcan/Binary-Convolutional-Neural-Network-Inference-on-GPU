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

constexpr std::pair<int, int> register_size(8, 4);

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
		T * __restrict__ d_odata,
        const size_t dim0,
        const size_t dim1,
        const size_t dim2)
{
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < (dim0*dim1)){
	  size_t tidx = idx;
	  T tsum = 0;
	  for (size_t i = 0; i < dim2; i++)
	  {
		tsum += d_idata[tidx];
		tidx += dim0*dim1;
	  }
	  d_odata[idx] = tsum / dim2;
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
		odata = pozitive<<(output_location[1] * register_size.second  + output_location[0]) | odata;
	}
	else
	{
		odata = negative<<(output_location[1] * register_size.second  + output_location[0]) | odata;
	}
}

template<typename T>
void __global__ to_binary_tensor(
	const T *  d_idata,
	unsigned int *  d_odata,
	const int row, const int b_row,
	const int col, const int b_col,
	const int channel = 1,
	const int kernel_row = 3, const int kernel_col = 3)
{
	// Each thread will store a size = 32 array inside their single register
	int idx = threadIdx.x+blockDim.x*blockIdx.x; //register IDX
	// n*(regsiter_size - kernel_size)
	if (idx < (b_row * b_col * channel))
	{
		int kernel_channel = idx / (b_row * b_col);
		int idx_matrix = idx % (b_row * b_col);
		int input_index[] = {(idx_matrix%b_col) * (register_size.first - kernel_col), (idx_matrix /b_col ) * (register_size.second - kernel_row), (kernel_channel)};
		int data_idx = input_index[0] + (input_index[1] + input_index[2] * col) * row;
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
				if (input_index[0] == row) break;
			}
			data_idx = data_idx + col;
			input_index[1] += 1;
			register_location[0] = 0;
			register_location[1] += 1;
			if (input_index[1] == col) break;
		}
		d_odata[idx] = local_register;
	}
}

int main()
{

	// Initialize a 3d tensor
	int row = 256;
	int col = 256;
	int channel = 16;

	int kernel_row = 3;
	int kernel_col = 3;
	int channel_in = 8;
	int channel_out = 16;



	// Test Channel Summation
	int *h_tensor = new int[channel * row * col];
	int *h_matrix = new int[row * col];
	int	*h_weights = new int[kernel_row * kernel_col * channel_in * channel_out];
	for (int i=0; channel * row * col>i; ++i) h_tensor[i] = (rand() % 100) + 100;
	//for (int i=0; kernel_row * kernel_col * channel_in * channel_out > i; ++i) (rand() % 100) - 50;

	// Channel summation
	assert( h_tensor != NULL);
	size_t sz = row * col * channel * sizeof(int);
	size_t rsz = row * col * sizeof(int);
	int *d_tensor;
	int *d_matrix;
	cudaError_t err = cudaMalloc(&d_tensor, sz);
	if (err != cudaSuccess) {printf("cudaMalloc1 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMalloc(&d_matrix, rsz);
	if (err != cudaSuccess) {printf("cudaMalloc2 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMemcpy(d_tensor, h_tensor, sz, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {printf("cudaMemset1 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMemset(d_matrix, 0 , rsz);
	if (err != cudaSuccess) {printf("cudaMemset2 error: %s\n", cudaGetErrorString(err)); return -1;}
	kernel_reduce_sum<<<((row * col)+(nTPB-1))/nTPB, nTPB>>>(d_tensor, d_matrix, col, row, channel);
	err = cudaMemcpy( h_matrix, d_matrix, rsz, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {printf("result1 error: %s\n", cudaGetErrorString(err)); return -1;}
	for (int i= 0; row>i; ++i)
	{
		for (int j=0; col>j; ++j)
		{
			std::cout<< h_matrix[i * col + j] << "  ";
		}
		std::cout<< std::endl;
	}
	std::cout << "Channel Sum" <<std::endl;
	// float|int to binary
	auto binary_size = to_binary_size(std::make_pair(col, row), std::make_pair(kernel_col, kernel_row));
	unsigned int *h_binary_tensor = new unsigned int[channel * binary_size.second * binary_size.first];
	unsigned *d_binary_tensor;
	size_t bsz =  binary_size.first * binary_size.second * channel * sizeof(unsigned int);
	cudaMalloc(&d_binary_tensor, bsz);
	size_t block_size = choose_block_size(binary_size.first * binary_size.second *channel);
	to_binary_tensor<int><<<(binary_size.first * binary_size.second * channel)/block_size + 1 , block_size>>>(d_tensor, d_binary_tensor, row, binary_size.second, col, binary_size.first, channel, kernel_row, kernel_col);
	cudaMemcpy(h_binary_tensor, d_binary_tensor, bsz, cudaMemcpyDeviceToHost);
	for (int k = 0; channel>k; ++k)
	{
		for (int i= 0; binary_size.second>i; ++i)
		{
			for (int j=0; binary_size.first>j; ++j)
			{
				std::cout<< h_binary_tensor[ (k * binary_size.second + i) * binary_size.first + j] << "  ";
			}
			std::cout<< std::endl;
		}
	}
	return 0;
}

