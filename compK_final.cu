
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include "stdio.h"
#include "iostream"
#include "math.h"
#include "chrono"

#define IM_WIDTH 1024
#define IM_HEIGHT 1024
#define IM_CHANNEL 64

#define BLOCKSIZE 64
#define TILE_WIDTH 64
#define maskCols 3
#define maskRows 3
#define w (TILE_WIDTH + maskCols -1)

typedef float type;



// chose convolution kernel type
// gl    -> global memory
// gl_co -> global and constant memory
// sh    -> shared memory
// sh_co -> shared and constant memory

std::string memory_type = "gl";

// to print results
bool debug = true;


//mask in constant memory
template <typename T>
__constant__ T deviceMaskData[maskRows * maskCols];


template <typename T>
__global__ void MeanMatrices(T* __restrict__ g_idata, T* __restrict__ g_odata, const int channels, const int width, const int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < height * width) {
        float sum = 0;
        for (int i = 0; i < channels; i++) {
            sum += g_idata[i * width * height + idx];
        }
        g_odata[idx] = sum / channels;
    }
}


template <typename T>
__global__ void convolution2D(T* InputImageData, T kernel_value,
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
                    else accum += 0;
                }

            }
            outputImageData[(row * width + col) * channels + k] = accum;
        }

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
    printf("\n");
    free(hostOutputImageData);
}


//using only global memory
template <typename T>
T* prepareInputGlobal(T* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels, float& elapsed_time_addition, float& elapsed_time_convolution) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = (imageHeight * imageWidth + BLOCKSIZE-1) / BLOCKSIZE;
    int convolutionChannels = 1;

    int numberBlockX = (imageWidth + BLOCKSIZE - 1) / BLOCKSIZE;
    int numberBlockY = (imageHeight + BLOCKSIZE - 1) / BLOCKSIZE;

    dim3 dimGrid(numberBlockX, numberBlockY);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);

    cudaEvent_t start1, stop1;
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);


    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMalloc((void**)&deviceConvOutputImageData, imageWidth * imageHeight *
        sizeof(T));

    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * sizeof(T));

    cudaEventRecord(start1, 0);

    MeanMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Addition Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        //print<T>(deviceOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }

    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));

    T kernel_value = 1.0 / (maskRows * maskCols);

    cudaEventRecord(start2, 0);

    convolution2D<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, kernel_value, deviceConvOutputImageData,
        convolutionChannels, imageWidth, imageHeight);


        

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsed_time_addition, start1, stop1);
    cudaEventElapsedTime(&elapsed_time_convolution, start2, stop2);


    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Convolution Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        //print<T>(deviceConvOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }
    return deviceConvOutputImageData;
}


int main() {

    int imageChannels = IM_CHANNEL;
    int imageHeight = IM_HEIGHT;
    int imageWidth = IM_WIDTH;

    type* hostInputImageData;
    type* deviceInputImageData;
    type* deviceOutputImageData;

    float elapsedTimeAddition, elapsedTimeConvolution;

    hostInputImageData = new type[imageHeight * imageWidth * imageChannels];

    // call only once in the main
    type hostMaskData[maskCols * maskRows];

    for (int i = 0; i < maskCols * maskRows; i++)
        hostMaskData[i] = static_cast<type>(1.0 / (maskRows * maskCols));

    cudaMemcpyToSymbol(deviceMaskData<type>, hostMaskData, maskCols * maskRows * sizeof(type));

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
    

    deviceOutputImageData = prepareInputGlobal<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels,
        elapsedTimeAddition, elapsedTimeConvolution);

    
    

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Timing Results" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "image dimensions (c x h x w):" << std::endl;
    std::cout << imageChannels << "x" << imageWidth << "x" << imageHeight << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    
    std::cout << "elapsed time for addition: " << elapsedTimeAddition << " ms" << std::endl;
    std::cout << "elapsed time for convolution: " << elapsedTimeConvolution << " ms" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    cudaMemset(deviceInputImageData, 0, imageWidth * imageHeight *
        imageChannels * sizeof(type));
    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight *
        sizeof(type));

    free(hostInputImageData);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

}







