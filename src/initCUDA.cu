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



// Texture reference for reading image
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

//// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
//#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)
//
//inline void __checkCudaErrors(cudaError err, const char *file, const int line)
//{
//    if (cudaSuccess != err)
//    {
//        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
//                file, line, (int)err, cudaGetErrorString(err));
//        exit(EXIT_FAILURE);
//    }
//}


__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale)
{
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));

    if (Sum < 0)
    {
        return 0;
    }
    else if (Sum > 0xff)
    {
        return 0xff;
    }

    return (unsigned char) Sum;
}

__global__ void
SobelShared(uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
            short BlockWidth, short SharedPitch,
#endif
            short w, short h, float fScale)
{
    short u = 4*blockIdx.x*BlockWidth;
    short v = blockIdx.y*blockDim.y + threadIdx.y;
    short ib;

    int SharedIdx = threadIdx.y * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x)
    {
        LocalBlock[SharedIdx+4*ib+0] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+0), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+1] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+1), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+2] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+2), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+3] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+3), (float)(v-RADIUS));
    }

    if (threadIdx.y < RADIUS*2)
    {
        //
        // copy trailing RADIUS*2 rows of pixels into shared
        //
        SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;

        for (ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x)
        {
            LocalBlock[SharedIdx+4*ib+0] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+0), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+1] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+1), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+2] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+2), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+3] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+3), (float)(v+blockDim.y-RADIUS));
        }
    }

    __syncthreads();

    u >>= 2;    // index as uchar4 from here
    uchar4 *pSobel = (uchar4 *)(((char *) pSobelOriginal)+v*SobelPitch);
    SharedIdx = threadIdx.y * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x)
    {

        unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
        unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
        unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
        unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
        unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
        unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
        unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
        unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
        unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

        uchar4 out;

        out.x = ComputeSobel(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = ComputeSobel(pix01, pix02, pix00,
                             pix11, pix12, pix10,
                             pix21, pix22, pix20, fScale);

        pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = ComputeSobel(pix02, pix00, pix01,
                             pix12, pix10, pix11,
                             pix22, pix20, pix21, fScale);

        pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = ComputeSobel(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        if (u+ib < w/4 && v < h)
        {
            pSobel[u+ib] = out;
        }
    }

    __syncthreads();
}

__global__ void
SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch,
               int w, int h, float fscale)
{
    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        pSobel[i] = min(max((tex2D(tex, (float) i, (float) blockIdx.x) * fscale), 0.f), 255.f);
    }
}

__global__ void
SobelTex(Pixel *pSobelOriginal, unsigned int Pitch,
         int w, int h, float fScale)
{
    unsigned char *pSobel =
        (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        unsigned char pix00 = tex2D(tex, (float) i-1, (float) blockIdx.x-1);
        unsigned char pix01 = tex2D(tex, (float) i+0, (float) blockIdx.x-1);
        unsigned char pix02 = tex2D(tex, (float) i+1, (float) blockIdx.x-1);
        unsigned char pix10 = tex2D(tex, (float) i-1, (float) blockIdx.x+0);
        unsigned char pix11 = tex2D(tex, (float) i+0, (float) blockIdx.x+0);
        unsigned char pix12 = tex2D(tex, (float) i+1, (float) blockIdx.x+0);
        unsigned char pix20 = tex2D(tex, (float) i-1, (float) blockIdx.x+1);
        unsigned char pix21 = tex2D(tex, (float) i+0, (float) blockIdx.x+1);
        unsigned char pix22 = tex2D(tex, (float) i+1, (float) blockIdx.x+1);
        pSobel[i] = ComputeSobel(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);
    }
}

//extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp)
//{
//    cudaChannelFormatDesc desc;
//
//    if (Bpp == 1)
//    {
//        desc = cudaCreateChannelDesc<unsigned char>();
//    }
//    else
//    {
//        desc = cudaCreateChannelDesc<uchar4>();
//    }
//
//    checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
//    checkCudaErrors(cudaMemcpyToArray(array, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice));
//}

extern "C" void deleteTexture(void)
{
    checkCudaErrors(cudaFreeArray(array));
}


// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode, float fScale)
{
    checkCudaErrors(cudaBindTextureToArray(tex, array));

    switch (mode)
    {
        case SOBELDISPLAY_IMAGE:
            SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih, fScale);
            break;
        case SOBELDISPLAY_SOBELTEX:
            SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale);
            break;
        case SOBELDISPLAY_SOBELSHARED:
            {
                dim3 threads(16,4);
#ifndef FIXED_BLOCKWIDTH
                int BlockWidth = 80; // must be divisible by 16 for coalescing
#endif
                dim3 blocks = dim3(iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
                                   ih/threads.y+(0!=ih%threads.y));
                int SharedPitch = ~0x3f&(4*(BlockWidth+2*RADIUS)+0x3f);
                int sharedMem = SharedPitch*(threads.y+2*RADIUS);

                // for the shared kernel, width must be divisible by 4
                iw &= ~3;

                SobelShared<<<blocks, threads, sharedMem>>>((uchar4 *) odata,
                                                            iw,
#ifndef FIXED_BLOCKWIDTH
                                                            BlockWidth, SharedPitch,
#endif
                                                            iw, ih, fScale);
            }
            break;
    }

    checkCudaErrors(cudaUnbindTexture(tex));
}


/********************************* AbstractKernel ***********************************************************/
void AbstractKernel::Reset(){
	cudaDeviceReset();
}

AbstractKernel::AbstractKernel():
		inputBufGpu(NULL), outputBufGpu(NULL)
{
	inBufWidth = inBufHeight = 0;
	outBufWidth = outBufHeight = 0;
	deviceIdx = 0;
//	checkCudaErrors(cudaSetDevice(deviceIdx));
}

AbstractKernel::~AbstractKernel(){
	if(inputBufGpu == outputBufGpu){
		if(inputBufGpu){
			checkCudaErrors(cudaFree(inputBufGpu));
			inputBufGpu = NULL;
			outputBufGpu = NULL;
		}
	}else{
		if(inputBufGpu){
			checkCudaErrors(cudaFree(inputBufGpu));
			inputBufGpu = NULL;
		}
		if(outputBufGpu){
			checkCudaErrors(cudaFree(outputBufGpu));
			outputBufGpu = NULL;
		}
	}
}

void AbstractKernel::SetDevice(int dev){
	deviceIdx = dev;
	checkCudaErrors(cudaSetDevice(deviceIdx));
}

void AbstractKernel::CopyToGPU(Image inputImgCPU){
	if( inputImgCPU.w*inputImgCPU.h >					// if memory was allocated, but less than need
    		inBufWidth*inBufHeight && inputBufGpu ){	// then realloc memory
			if(inputBufGpu == outputBufGpu)
				outputBufGpu = NULL;
			checkCudaErrors(cudaFree(inputBufGpu));
			inputBufGpu = NULL;
	}
	if(!inputBufGpu)
		checkCudaErrors(cudaMalloc(&inputBufGpu, sizeof(Pixel)*inputImgCPU.w*inputImgCPU.h));
	checkCudaErrors(cudaMemcpy(inputBufGpu, inputImgCPU.data,
			sizeof(Pixel)*inputImgCPU.w*inputImgCPU.h, cudaMemcpyHostToDevice));
	outputBufGpu = inputBufGpu;
	inBufWidth = inBufWidth = inputImgCPU.w;
	outBufHeight = inBufHeight = inputImgCPU.h;
}

void AbstractKernel::CopyFromGPU(Image *outputImgCPU){
	outputImgCPU->w = outBufWidth;
	outputImgCPU->h = outBufHeight;
	checkCudaErrors(cudaMemcpy(outputImgCPU->data, outputBufGpu,
			sizeof(Pixel)*outputImgCPU->w*outputImgCPU->h, cudaMemcpyHostToDevice));
}

/********************************* SobelKernel ***********************************************************/
SobelKernel::SobelKernel(){
	desc = cudaCreateChannelDesc<unsigned char>();
}

SobelKernel::~SobelKernel(){
	deleteTexture();
}

void SobelKernel::SetPropeties(enum SobelDisplayMode mode, float fScale){
	sobelDisplayMode = mode;
	imageScale = fScale;
}

void SobelKernel::CopyToGPU(Image inputImgCPU){
//	if (pixSize == 1) {
//		desc = cudaCreateChannelDesc<unsigned char>();
//	} else {
//		desc = cudaCreateChannelDesc<uchar4>();
//	}
//	checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
//	checkCudaErrors(cudaMemcpyToArray(array, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice));
	inBufWidth = outBufWidth = inputImgCPU.w;
	inBufHeight = outBufHeight = inputImgCPU.h;
//	setupTexture(inputImgCPU.w, inputImgCPU.h, inputImgCPU.data, pixSize);
	if(array != NULL){
		checkCudaErrors(cudaFreeArray(array));
		array = NULL;
	}
	printf("Alocate memory %dx%d\n", inBufWidth, inBufHeight);

	checkCudaErrors(cudaMallocArray(&array, &desc, inBufWidth, inBufHeight));
	checkCudaErrors(cudaMemcpyToArray(array, 0, 0, inputImgCPU.data,
			sizeof(Pixel)*inBufWidth*inBufHeight, cudaMemcpyHostToDevice));
}


Pixel* SobelKernel::CallKernel(Pixel *outputBUFFER){
//	outputBufGpu = outputBUFFER;

//	sobelFilter(outputBUFFER, outBufWidth, outBufHeight, sobelDisplayMode, imageScale);
	checkCudaErrors(cudaMemcpyFromArray(outputBUFFER, array, 0,0, sizeof(Pixel)*inBufWidth*inBufHeight, cudaMemcpyDeviceToDevice));
	return outputBUFFER;
}

/********************************* MalvarKernel ***********************************************************/
MalvarKernel::MalvarKernel():
	BLOCKX(128), BLOCKY(16)
{}

void MalvarKernel::CopyToGPU(Image inputImgCPU){
	cudaSetDeviceFlags(cudaDeviceMapHost);
	dimBlock = dim3(128);
	dimGrid = dim3((inputImgCPU.w-4 + BLOCKX - 1) / BLOCKX, (inputImgCPU.h - 4 + BLOCKY - 1) /BLOCKY);
}

Pixel* MalvarKernel::CallKernel(Pixel *outputBUFFER){

	return outputBufGpu;
}

/********************************* MalvarKernel ***********************************************************/
PitchKernel::PitchKernel(){
	imPitch = 100;
}

void PitchKernel::CopyToGPU(Image inputImgCPU){
	inBufWidth = inputImgCPU.w;
	imPitch += inBufWidth;
	inBufHeight = inputImgCPU.h;

	if (inputBufGpu != NULL) {
		checkCudaErrors(cudaFree(inputBufGpu));
		inputBufGpu = NULL;
	}
	printf("Alocate memory %dx%d\n", inBufWidth, inBufHeight);
	checkCudaErrors(cudaMalloc(&inputBufGpu, inBufHeight*(inBufWidth + imPitch)*sizeof(Pixel)));
//	for(int i=0; i < inBufHeight; i++){
		checkCudaErrors(cudaMemcpy2D( inputBufGpu, imPitch,
				inputImgCPU.data, inputImgCPU.w,
				inBufWidth, inBufHeight,
				cudaMemcpyHostToDevice ));
//	}
}

void PitchKernel::GetActualSize(int *width, int *height, int *pitch){
	*width = inBufWidth;
	*height = inBufHeight;
	*pitch = imPitch;
}

Pixel* PitchKernel::CallKernel(Pixel *outputBUFFER){
	checkCudaErrors(cudaMemcpy2D( outputBUFFER, imPitch,
			inputBufGpu, imPitch,
			inBufWidth, inBufHeight,
			cudaMemcpyHostToDevice ));

	return outputBUFFER;
}
