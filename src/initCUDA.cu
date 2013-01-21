/*
 * initCUDA.cu
 *
 *  Created on: 27.12.2012
 *      Author: id23cat
 */
#include "initCUDA.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>



// pointer and properties for line memory
Pixel *devImage;
int imWidth;
int imHeight;
int imPitch;

Pixel *loadToDevice(int width, int height, Pixel *data, int pixelsize)
{
	checkCudaErrors(cudaMalloc(&devImage, width*height*pixelsize));
	checkCudaErrors(cudaMemcpy(devImage, data, width*height*pixelsize, cudaMemcpyHostToDevice));
	imWidth = width;
	imHeight = height;
	imPitch = 0;

	return devImage;
}

void deleteImage(Pixel *devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}
