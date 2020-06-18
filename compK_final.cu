
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include "stdio.h"
#include "iostream"
#include "math.h"
#include "chrono"

#define BLOCKSIZE 32
#define TILE_WIDTH 16
#define maskCols 3
#define maskRows 3
#define w (TILE_WIDTH + maskCols -1)

#define type float


//mask in constant memory
template <typename T>
__constant__ T deviceMaskData;


template <typename T>
__global__ void prepareInputGlobalKernel(T* InputImageData, T kernel_value,
    T* outputImageData, int channels, int width, int height) {

    float accum;
    int col = threadIdx.x + blockIdx.x * blockDim.x;   //col index
    int row = threadIdx.y + blockIdx.y * blockDim.y;   //row index
    int maskRowsRadius = maskRows / 2;
    int maskColsRadius = maskCols / 2;


    for (int k = 0; k < channels; k++) {      //cycle on kernel channels
        if (row < height && col < width) {
            accum = 0;
            int startRow = row - maskRowsRadius;  //row index shifted by mask radius
            int startCol = col - maskColsRadius;  //col index shifted by mask radius

            for (int i = 0; i < maskRows; i++) { //cycle on mask rows

                for (int j = 0; j < maskCols; j++) { //cycle on mask columns

                    int currentRow = startRow + i; // row index to fetch data from input image
                    int currentCol = startCol + j; // col index to fetch data from input image

                    if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {

                        accum += InputImageData[(currentRow * width + currentCol) * channels + k] *
                            kernel_value;
                    }
                    else accum = 0;
                }

            }
            outputImageData[(row * width + col) * channels + k] = accum;
        }

    }
}


template <typename T>
__global__ void prepareInputGlobalConstantKernel(T* InputImageData,
    T* outputImageData, int channels, int width, int height) {

    float accum;
    int col = threadIdx.x + blockIdx.x * blockDim.x;   //col index
    int row = threadIdx.y + blockIdx.y * blockDim.y;   //row index
    int maskRowsRadius = maskRows / 2;
    int maskColsRadius = maskCols / 2;


    for (int k = 0; k < channels; k++) {      //cycle on kernel channels
        if (row < height && col < width) {
            accum = 0;
            int startRow = row - maskRowsRadius;  //row index shifted by mask radius
            int startCol = col - maskColsRadius;  //col index shifted by mask radius

            for (int i = 0; i < maskRows; i++) { //cycle on mask rows

                for (int j = 0; j < maskCols; j++) { //cycle on mask columns

                    int currentRow = startRow + i; // row index to fetch data from input image
                    int currentCol = startCol + j; // col index to fetch data from input image

                    if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {

                        accum += InputImageData[(currentRow * width + currentCol) * channels + k] *
                            deviceMaskData<T>;
                    }
                    else accum = 0;
                }

            }
            outputImageData[(row * width + col) * channels + k] = accum;
        }

    }
}


template <typename T>
__global__ void prepareInputSharedKernel(T* InputImageData,
    T* outputImageData, T kernel_value, int channels, int width, int height) {

    __shared__ T N_ds[w][w];	//block of share memory


    // allocation in shared memory of image blocks
    int maskRadius = maskRows / 2;
    for (int k = 0; k < channels; k++) {
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;     //col of shared memory
        int destX = dest % w;		//row of shared memory
        int srcY = blockIdx.y * TILE_WIDTH + destY - maskRadius;  //row index to fetch data from input image
        int srcX = blockIdx.x * TILE_WIDTH + destX - maskRadius;	//col index to fetch data from input image
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = InputImageData[(srcY * width + srcX) * channels + k];
        else
            N_ds[destY][destX] = 0;


        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - maskRadius;
        srcX = blockIdx.x * TILE_WIDTH + destX - maskRadius;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = InputImageData[(srcY * width + srcX) * channels + k];
            else
                N_ds[destY][destX] = 0;
        }

        __syncthreads();


        //compute kernel convolution
        float accum = 0;
        int y, x;
        for (y = 0; y < maskCols; y++)
            for (x = 0; x < maskRows; x++)
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * kernel_value;

        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            outputImageData[(y * width + x) * channels + k] = accum;
        __syncthreads();

    }
}


template <typename T>
__global__ void prepareInputSharedConstantKernel(T* InputImageData,
    T* outputImageData, int channels, int width, int height) {

    __shared__ T N_ds[w][w];	//block of share memory


    // allocation in shared memory of image blocks
    int maskRadius = maskRows / 2;
    for (int k = 0; k < channels; k++) {
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;     //col of shared memory
        int destX = dest % w;		//row of shared memory
        int srcY = blockIdx.y * TILE_WIDTH + destY - maskRadius;  //row index to fetch data from input image
        int srcX = blockIdx.x * TILE_WIDTH + destX - maskRadius;	//col index to fetch data from input image
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = InputImageData[(srcY * width + srcX) * channels + k];
        else
            N_ds[destY][destX] = 0;


        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - maskRadius;
        srcX = blockIdx.x * TILE_WIDTH + destX - maskRadius;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = InputImageData[(srcY * width + srcX) * channels + k];
            else
                N_ds[destY][destX] = 0;
        }

        __syncthreads();


        //compute kernel convolution
        float accum = 0;
        int y, x;
        for (y = 0; y < maskCols; y++)
            for (x = 0; x < maskRows; x++)
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * deviceMaskData<T>;

        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            outputImageData[(y * width + x) * channels + k] = accum;
        __syncthreads();

    }
}


template <typename T>
__global__ void addMatrices(T* matrices, T* result, int channels, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < height * width) {
        float sum = 0;
        for (int i = 0; i < channels; i++) {
            sum += matrices[i * width * height + idx];
        }
        result[idx] = sum/channels;
    }
}


template <typename T>
void clearMemory(T* p, int size) {
    cudaMemset(p, 0, size);
    cudaFree(p);
}


template <typename T>
void print(T* deviceOutputImageData, int imageHeight, int imageWidth) {
    T* hostOutputImageData;
    hostOutputImageData = new T[imageHeight * imageWidth];
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
    imageWidth * imageHeight * sizeof(T),
    cudaMemcpyDeviceToHost);

    for (int i = 0; i < imageHeight * imageWidth; i++) {
        printf("%f, ", hostOutputImageData[i]);
    }

    free(hostOutputImageData);
}


//using only global memory
template <typename T>
T* prepareInputGlobal(float* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = imageHeight * imageWidth / BLOCKSIZE;
    int convolutionChannels = 1;

    float numberBlockXTiling = (float)imageWidth / TILE_WIDTH;
    float numberBlockYTiling = (float)imageHeight / TILE_WIDTH;

    int numberBlockX = ceil(numberBlockXTiling);
    int numberBlockY = ceil(numberBlockYTiling);

    dim3 dimGrid(numberBlockX, numberBlockY);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMalloc((void**)&deviceConvOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * sizeof(T));


    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    //print<T>(deviceOutputImageData, imageHeight, imageWidth);


    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));

    T kernel_value = 1.0 / (maskRows * maskCols);

    prepareInputGlobalKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, kernel_value, deviceConvOutputImageData,
        convolutionChannels, imageWidth, imageHeight);


    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    return deviceConvOutputImageData;
}


// using global memory and constant memory
template <typename T>
T* prepareInputGlobalConstant(float* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = imageHeight * imageWidth / BLOCKSIZE;
    int convolutionChannels = 1;

    float numberBlockXTiling = (float)imageWidth / TILE_WIDTH;
    float numberBlockYTiling = (float)imageHeight / TILE_WIDTH;

    int numberBlockX = ceil(numberBlockXTiling);
    int numberBlockY = ceil(numberBlockYTiling);

    dim3 dimGrid(numberBlockX, numberBlockY);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMalloc((void**)&deviceConvOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * sizeof(T));


    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    //print<T>(deviceOutputImageData, imageHeight, imageWidth);


    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));


    prepareInputGlobalConstantKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, deviceConvOutputImageData,
        convolutionChannels, imageWidth, imageHeight);


    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    return deviceConvOutputImageData;
}


//using only shared memory
template <typename T>
float* prepareInputShared(float* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = imageHeight * imageWidth / BLOCKSIZE;
    int convolutionChannels = 1;

    float numberBlockXTiling = (float)imageWidth / TILE_WIDTH;
    float numberBlockYTiling = (float)imageHeight / TILE_WIDTH;

    int numberBlockX = ceil(numberBlockXTiling);
    int numberBlockY = ceil(numberBlockYTiling);

    dim3 dimGrid(numberBlockX, numberBlockY);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMalloc((void**)&deviceConvOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * sizeof(T));


    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    //print<T>(deviceOutputImageData, imageHeight, imageWidth);


    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));

    T kernel_value = 1.0 / (maskRows * maskCols);
    prepareInputSharedKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, deviceConvOutputImageData, kernel_value,
        convolutionChannels, imageWidth, imageHeight);


    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    return deviceConvOutputImageData;
}


//using constant and shared memory
template <typename T>
T* prepareInputSharedConstant(float* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = imageHeight * imageWidth / BLOCKSIZE;
    int convolutionChannels = 1;

    float numberBlockXTiling = (float)imageWidth / TILE_WIDTH;
    float numberBlockYTiling = (float)imageHeight / TILE_WIDTH;

    int numberBlockX = ceil(numberBlockXTiling);
    int numberBlockY = ceil(numberBlockYTiling);

    dim3 dimGrid(numberBlockX, numberBlockY);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMalloc((void**)&deviceConvOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * sizeof(T));


    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    //print<T>(deviceOutputImageData, imageHeight, imageWidth);


    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));


    prepareInputSharedConstantKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, deviceConvOutputImageData,
        convolutionChannels, imageWidth, imageHeight);


    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    return deviceConvOutputImageData;
}

int main() {

    int imageChannels = 64;
    int imageHeight = 8;
    int imageWidth = 8;

    type* hostInputImageData;
    type* deviceInputImageData;
    type* deviceOutputImageData;

    hostInputImageData = new type[imageHeight * imageWidth * imageChannels];

    // call only once in the main
    type hostMaskData;

    hostMaskData = 1.0 / (maskRows * maskCols);

    cudaMemcpyToSymbol(&deviceMaskData<type>, &hostMaskData, 1 * sizeof(type));

    for (int i = 0; i < imageChannels * imageHeight * imageWidth; i++) {
        hostInputImageData[i] = type((float)rand() / (RAND_MAX));
    }

    //for (int i = 0; i < imageChannels * imageHeight * imageWidth; i++) {
    //    printf("%f, ", hostInputImageData[i]);
    //}

    std::cout << "\n" << std::endl;

    cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(type));

    cudaMemcpy(deviceInputImageData, hostInputImageData,
        imageWidth * imageHeight * imageChannels * sizeof(type),
        cudaMemcpyHostToDevice);
    
    std::chrono::high_resolution_clock::time_point startGlobal = std::chrono::high_resolution_clock::now();
    deviceOutputImageData = prepareInputGlobal<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels);
    std::chrono::high_resolution_clock::time_point endGlobal = std::chrono::high_resolution_clock::now();

    //std::chrono::high_resolution_clock::time_point startGlobalConstant = std::chrono::high_resolution_clock::now();
    //deviceOutputImageData = prepareInputGlobalConstant<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels);
    //std::chrono::high_resolution_clock::time_point endGlobalConstant = std::chrono::high_resolution_clock::now();

    //std::chrono::high_resolution_clock::time_point startShared = std::chrono::high_resolution_clock::now();
    //deviceOutputImageData = prepareInputShared<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels);
    //std::chrono::high_resolution_clock::time_point endShared = std::chrono::high_resolution_clock::now();

    //std::chrono::high_resolution_clock::time_point startSharedConstant = std::chrono::high_resolution_clock::now();
    //deviceOutputImageData = prepareInputSharedConstant<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels);
    //std::chrono::high_resolution_clock::time_point endSharedConstant = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double>  durationGlobal = endGlobal - startGlobal;
    //std::chrono::duration<double>  durationGlobalConstant = endGlobalConstant - startGlobalConstant;
    //std::chrono::duration<double>  durationShared = endShared - startShared;
    //std::chrono::duration<double>  durationSharedConstant = endSharedConstant - startSharedConstant;
    
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Timing Results" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "image dimensions (c x h x w):" << std::endl;
    std::cout << imageChannels << "x" << imageWidth << "x" << imageHeight << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    
    std::cout << "Using global memory" << std::endl;
    std::cout << "elapsed in time: " << durationGlobal.count()*1000 <<std::endl;
    std::cout << "-------------------------------------" << std::endl;

    //std::cout << "Using global and constant memory" << std::endl;
    //std::cout << "elapsed in time: " << durationGlobalConstant.count() * 1000 << std::endl;
    //std::cout << "-------------------------------------" << std::endl;

    //std::cout << "Using shared memory" << std::endl;
    //std::cout << "elapsed in time: " << durationShared.count() * 1000 << std::endl;
    //std::cout << "-------------------------------------" << std::endl;

    //std::cout << "Using shared and constant memory" << std::endl;
    //std::cout << "elapsed in time: " << durationSharedConstant.count() * 1000 << std::endl;
    //std::cout << "-------------------------------------" << std::endl;


    cudaMemset(deviceInputImageData, 0, imageWidth * imageHeight *
        imageChannels * sizeof(type));
    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight *
        sizeof(type));

    free(hostInputImageData);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

}