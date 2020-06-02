#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <vector>
#include <assert.h>

constexpr std::pair<int, int> register_size(4, 8);

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
	  d_odata[idx] = tsum;
	}
}

int main()
{

	// Initialize a 3d tensor
	int row = 5;
	int col = 5;
	int channel = 5;

	int*** tensor_ptr = new int**[channel];
	for (int i= 0; channel>i; ++i)
	{
		tensor_ptr[i] = new int*[row];
		for (int j= 0; row>j; ++j)
		{
			tensor_ptr[i][j] = new int[col];
			for (int k= 0; col>k; ++k)
			{
				tensor_ptr[i][j][k] = (rand() % 100) - 50;
			}
		}
	}
	int ** matrix_ptr = new int*[row];
	for (int j= 0; row>j; ++j)
	{
		matrix_ptr[j] = new int[col];
		for (int k= 0; col>k; ++k)
		{
			matrix_ptr[j][k] = 0;
		}
	}
	Matrix3d<int> *h_tensor;
	Matrix2d<int> *h_matrix;

	h_tensor = new Matrix3d<int>(channel, row, col, tensor_ptr);
	h_matrix = new Matrix2d<int>(row, col, matrix_ptr);

	// Test Channel Summation

	assert(h_tensor->tensor!= NULL);
	size_t sz = h_tensor->row * h_tensor->col * h_tensor->channel * sizeof(int);
	size_t rsz = h_tensor->row * h_tensor->col * sizeof(int);
	int *d_tensor;
	int *d_matrix;
	cudaError_t err = cudaMalloc(&d_tensor, sz);
	if (err != cudaSuccess) {printf("cudaMalloc1 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMalloc(&d_matrix, rsz);
	if (err != cudaSuccess) {printf("cudaMalloc2 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMemcpy(d_tensor, h_tensor->tensor, sz, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {printf("cudaMemset1 error: %s\n", cudaGetErrorString(err)); return -1;}
	err = cudaMemset(d_matrix, 0 , rsz);
	if (err != cudaSuccess) {printf("cudaMemset2 error: %s\n", cudaGetErrorString(err)); return -1;}
	kernel_reduce_sum<<<((h_tensor->row * h_tensor->col)+(nTPB-1))/nTPB, nTPB>>>(d_tensor, d_matrix, h_tensor->col, h_tensor->row, h_tensor->channel);
	err = cudaMemcpy(h_matrix->data, d_matrix, rsz, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {printf("result1 error: %s\n", cudaGetErrorString(err)); return -1;}
	for (int i= 0; row>i; ++i)
	{
		for (int j=0; col>j; ++j)
		{
			std::cout<< h_matrix->data[i][j] << "  ";
		}
		std::cout<< std::endl;
	}
	return 0;
}

