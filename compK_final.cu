
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
#define type float



// chose convolution kernel type
// gl    -> global memory
// gl_co -> global and constant memory
// sh    -> shared memory
// sh_co -> shared and constant memory

std::string memory_type = "gl";

// to print results
bool debug = false;


//mask in constant memory
template <typename T>
__constant__ T deviceMaskData[maskRows * maskCols];


template <typename T>
__global__ void addMatrices(T* __restrict__ g_idata, T* __restrict__ g_odata, const int channels, const int width, const int height)
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
                    else accum += 0;
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
                            deviceMaskData<type>[i * maskRows + j];
                    }
                    else accum += 0;
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
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * deviceMaskData<type>[y * maskCols + x];;

        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            outputImageData[(y * width + x) * channels + k] = accum;
        __syncthreads();

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

    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Addition Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }

    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));

    T kernel_value = 1.0 / (maskRows * maskCols);

    cudaEventRecord(start2, 0);

    prepareInputGlobalKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, kernel_value, deviceConvOutputImageData,
        convolutionChannels, imageWidth, imageHeight);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Convolution Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceConvOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }
        

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsed_time_addition, start1, stop1);
    cudaEventElapsedTime(&elapsed_time_convolution, start2, stop2);


    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    return deviceConvOutputImageData;
}


// using global memory and constant memory
template <typename T>
T* prepareInputGlobalConstant(T* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels, float& elapsed_time_addition, float& elapsed_time_convolution) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = (imageHeight * imageWidth + BLOCKSIZE - 1) / BLOCKSIZE;
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

    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Addition Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }

    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));

    cudaEventRecord(start2, 0);

    prepareInputGlobalConstantKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, deviceConvOutputImageData,
        convolutionChannels, imageWidth, imageHeight);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsed_time_addition, start1, stop1);
    cudaEventElapsedTime(&elapsed_time_convolution, start2, stop2);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Convolution Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceConvOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }
        

    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    return deviceConvOutputImageData;
}


//using only shared memory
template <typename T>
float* prepareInputShared(T* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels, float& elapsed_time_addition, float& elapsed_time_convolution) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = (imageWidth * imageHeight + (BLOCKSIZE - 1)) / BLOCKSIZE;
    int convolutionChannels = 1;

    int numberBlockX = (imageWidth + (TILE_WIDTH - 1)) / TILE_WIDTH;
    int numberBlockY = (imageHeight + (TILE_WIDTH - 1)) / TILE_WIDTH;

    dim3 dimGrid(numberBlockX, numberBlockY);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

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

    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);

    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Addition Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }


    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));

    T kernel_value = 1.0 / (maskRows * maskCols);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    prepareInputSharedKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, deviceConvOutputImageData, kernel_value,
        convolutionChannels, imageWidth, imageHeight);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsed_time_addition, start1, stop1);
    cudaEventElapsedTime(&elapsed_time_convolution, start2, stop2);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Convolution Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceConvOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }

    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

    return deviceConvOutputImageData;
}


//using constant and shared memory
template <typename T>
T* prepareInputSharedConstant(T* deviceInputImageData, int imageHeight, int imageWidth, int imageChannels, float &elapsed_time_addition, float& elapsed_time_convolution) {

    T* deviceOutputImageData;
    T* deviceConvOutputImageData;

    int gridSize = (imageWidth * imageHeight + (BLOCKSIZE - 1)) / BLOCKSIZE;
    int convolutionChannels = 1;

    int numberBlockX = (imageWidth + (TILE_WIDTH-1)) / TILE_WIDTH;
    int numberBlockY = (imageHeight + (TILE_WIDTH-1)) / TILE_WIDTH;


    dim3 dimGrid(numberBlockX, numberBlockY);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

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
    
    addMatrices<T> << <gridSize, BLOCKSIZE >> > (deviceInputImageData, deviceOutputImageData,
        imageChannels, imageWidth, imageHeight);
    
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Addition Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }
    
    clearMemory<T>(deviceInputImageData, imageWidth * imageHeight *
        imageChannels * sizeof(T));
  
    cudaEventRecord(start2, 0);

    prepareInputSharedConstantKernel<T> << <dimGrid, dimBlock >> > (deviceOutputImageData, deviceConvOutputImageData,
        convolutionChannels, imageWidth, imageHeight);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    cudaEventElapsedTime(&elapsed_time_addition, start1, stop1);
    cudaEventElapsedTime(&elapsed_time_convolution, start2, stop2);

    if (debug == true) {
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Convolution Results" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        print<T>(deviceConvOutputImageData, imageHeight, imageWidth);
        std::cout << "-------------------------------------" << std::endl;
    }

    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    clearMemory(deviceOutputImageData, imageWidth * imageHeight * sizeof(T));

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
    
    if (memory_type.compare("gl") == 0 )  {
        deviceOutputImageData = prepareInputGlobal<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels,
            elapsedTimeAddition, elapsedTimeConvolution);
    }
    
    else if (memory_type.compare("gl_co") == 0) {
        deviceOutputImageData = prepareInputGlobalConstant<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels,
            elapsedTimeAddition, elapsedTimeConvolution);
    }
    

    else if (memory_type.compare("sh") == 0) {
        deviceOutputImageData = prepareInputShared<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels,
            elapsedTimeAddition, elapsedTimeConvolution);
    }
   

    else if (memory_type.compare("sh_co") == 0) {
        deviceOutputImageData = prepareInputSharedConstant<type>(deviceInputImageData, imageHeight, imageWidth, imageChannels,
            elapsedTimeAddition, elapsedTimeConvolution);
    }
    
    

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Timing Results" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "image dimensions (c x h x w):" << std::endl;
    std::cout << imageChannels << "x" << imageWidth << "x" << imageHeight << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    
    if (memory_type.compare("gl") == 0) {
        std::cout << "Using global memory" << std::endl;
        std::cout << "elapsed time for addition: " << elapsedTimeAddition << " ms" << std::endl;
        std::cout << "elapsed time for convolution: " << elapsedTimeConvolution << " ms" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
    }
    
    else if (memory_type.compare("gl_co") == 0) {
        std::cout << "Using global and constant memory" << std::endl;
        std::cout << "elapsed time for addition: " << elapsedTimeAddition << " ms" << std::endl;
        std::cout << "elapsed time for convolution: " << elapsedTimeConvolution << " ms" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
    }
    

    else if (memory_type.compare("sh") == 0) {
        std::cout << "Using shared memory" << std::endl;
        std::cout << "elapsed time for addition: " << elapsedTimeAddition << " ms" << std::endl;
        std::cout << "elapsed time for convolution: " << elapsedTimeConvolution << " ms" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
    }
    

    else if (memory_type.compare("sh_co") == 0) {
        std::cout << "Using shared and constant memory" << std::endl;
        std::cout << "elapsed time for addition: " << elapsedTimeAddition << " ms" << std::endl;
        std::cout << "elapsed time for convolution: " << elapsedTimeConvolution << " ms" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
    }
    


    cudaMemset(deviceInputImageData, 0, imageWidth * imageHeight *
        imageChannels * sizeof(type));
    cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight *
        sizeof(type));

    free(hostInputImageData);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

}







