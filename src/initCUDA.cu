/*
 * initCUDA.cu
 *
 *  Created on: 27.12.2012
 *      Author: id23cat
 */
#include "initCUDA.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking


// Texture reference for reading image
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

#define RADIUS 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

inline __device__ unsigned char clamp(int value){
	unsigned int v;
	asm("cvt.sat.u8.s32 %0, %1;" : "=r"(v) : "r"(value));
	return v;
}

inline __device__ int abs_i(int value){
	unsigned int v;
	asm("abs.s32 %0, %1;" : "=r"(v) : "r"(value));
	return v;
}

template <size_t BLOCKSIZEX, size_t BLOCKSIZEY> __device__ void LoadGlobalToSM_g(unsigned int* input, size_t ibrw, unsigned char* smem )
{
    const size_t inputBlockW = BLOCKSIZEX + 4;
    const size_t inputBlockW_4 = inputBlockW>>2;
    const size_t BLOCKSIZEX4 = BLOCKSIZEX/4;

    size_t ibrw4 = ibrw>>2;

    int warpId = threadIdx.x>>5;
    int threadId = threadIdx.x & 31;
    const int warpDim = blockDim.x>>5;

    unsigned int* input_t = input + blockIdx.y*BLOCKSIZEY*ibrw4 + warpId*ibrw4 + blockIdx.x*BLOCKSIZEX4 + threadId;
    unsigned int* inputBlock = (unsigned int*)smem + warpId*inputBlockW_4 + threadId;
    
	#pragma unroll
	for (int i=0;i<4;i++)
	{
		*inputBlock = *input_t;
		if (threadId==0)
			*(inputBlock+32)= *(input_t+32);

		inputBlock += warpDim*inputBlockW_4;
        input_t += warpDim*ibrw4;
	}

    if (warpId<4)
    {
        *inputBlock = *input_t;
        if (threadId==0)
            *(inputBlock+32)= *(input_t+32);
    }
}

template <size_t BLOCKSIZEX, size_t BLOCKSIZEY> __device__ void LoadGlobalToSM_hv(unsigned int* input, size_t ibrw, unsigned char* smem, size_t inputImgW, size_t inputImgH )
{
    const size_t inputBlockW = BLOCKSIZEX + 4;
    const size_t inputBlockW_4 = inputBlockW>>2;
    const size_t BLOCKSIZEX4 = BLOCKSIZEX/4;

    size_t ibrw4 = ibrw>>2;

    int warpId = threadIdx.x>>5;
    int threadId = threadIdx.x & 31;
    int warpDim = blockDim.x>>5;

    const size_t RealBlockWidth = inputImgW - blockIdx.x*BLOCKSIZEX;
    const size_t RealBlockWidthWord = (RealBlockWidth+3)/4;
    const size_t RealBlockHeight = inputImgH - blockIdx.y*BLOCKSIZEY;

    unsigned int* input_t = input + blockIdx.y*BLOCKSIZEY*ibrw4 + warpId*ibrw4 + blockIdx.x*BLOCKSIZEX4 + threadId;
    unsigned int* inputBlock = (unsigned int*)smem + warpId*inputBlockW_4 + threadId;

    
	#pragma unroll
	for (int i=0;i<4;i++)
	{
		if (threadId<RealBlockWidthWord && warpDim*i + warpId< RealBlockHeight)
			*inputBlock = *input_t;
		if (threadId==0 && 32<RealBlockWidthWord && warpDim*i + warpId< RealBlockHeight) 
			*(inputBlock+32)= *(input_t+32);

		input_t += warpDim*ibrw4;
        inputBlock += warpDim*inputBlockW_4;
	}

    if (warpId<4 && warpDim*4 + warpId< RealBlockHeight && threadId<RealBlockWidthWord) 
    {
        *inputBlock = *input_t;
        if (threadId==0 && 32<RealBlockWidthWord)
            *(inputBlock+32)= *(input_t+32);
    }
}

template <size_t BLOCKSIZEX, size_t BLOCKSIZEY> __device__ void WriteSMtoGlobal_g(unsigned int* output, size_t obrw,unsigned char* smem )
{
    int warpId = threadIdx.x>>5;
    int threadId = threadIdx.x & 31;

	const size_t obrw4 = (obrw*3)>>2;
	const int oRowWidth = BLOCKSIZEX * 3;
	const int oRowWidth4 = oRowWidth>>2;
	const int BLOCKSIZEX4 = BLOCKSIZEX>>2;
	unsigned int *ob_t = (unsigned int*)(smem + (BLOCKSIZEX + 4)*(BLOCKSIZEY + 4) + warpId*oRowWidth*4) + threadId;
    unsigned int *output_t =output + 2*obrw4 + (blockIdx.y*BLOCKSIZEY+ 4*warpId)*obrw4 + blockIdx.x*oRowWidth4 + threadId /*+3*/;

	#pragma unroll 
    for(int i=0;i<4;i++)
    {   
		*output_t = *ob_t;
		output_t[32] = ob_t[32];
		output_t[64] = ob_t[64];    
        output_t += obrw4;
        ob_t += oRowWidth4;
    }
}

template <size_t BLOCKSIZEX, size_t BLOCKSIZEY> __device__ void  WriteSMtoGlobal_hv(unsigned int* output, size_t obrw,unsigned char* smem, size_t inputImgW, size_t inputImgH )
{
	int warpId = threadIdx.x>>5;
    int threadId = threadIdx.x & 31;

	const size_t obrw4 = (obrw*3)>>2;;
	const int oRowWidth = BLOCKSIZEX * 3;
	const int oRowWidth4 = oRowWidth>>2;
	const int BLOCKSIZEX4 = BLOCKSIZEX>>2;

	unsigned int *ob_t = (unsigned int*)(smem + (BLOCKSIZEX + 4)*(BLOCKSIZEY + 4) + warpId*oRowWidth*4) + threadId;
    unsigned int *output_t =output + 2*obrw4 + /*3 +*/ (blockIdx.y*BLOCKSIZEY+ 4*warpId)*obrw4 + blockIdx.x*oRowWidth4 + threadId;

	size_t RealBlockWidth = inputImgW - blockIdx.x*BLOCKSIZEX - 4;
    const size_t RealBlockHeight = inputImgH - blockIdx.y*BLOCKSIZEY - 4;
	size_t RealBlockWidth4 = (RealBlockWidth*3+3)>>2;

	if (4*warpId>=RealBlockHeight)
		return;

    #pragma unroll 
    for(int i=0;i<4;i++)
    { 
		if (threadId<RealBlockWidth4)
			*output_t = *ob_t;
		if (threadId+32<RealBlockWidth4)
			output_t[32] = ob_t[32];
		if (threadId+64<RealBlockWidth4)
			output_t[64] = ob_t[64];    

		if (4*warpId + i>=RealBlockHeight)
			break;
        output_t += obrw4;
        ob_t += oRowWidth4;
    }
}

template <size_t BLOCKSIZEX, size_t BLOCKSIZEY> __device__ void Malvar_g (unsigned char* smem )
{
    const int threadId = threadIdx.x & 31;
	const int rx = threadIdx.x & 15;
	const int ry = threadIdx.x>>4;
	const int iRowWidth = BLOCKSIZEX + 4;
	const int iRowWidth2 = iRowWidth*2;
	const int iRowWidth3 = iRowWidth*3;
	const int oRowWidth = BLOCKSIZEX * 3;

	unsigned char *ib = smem + (ry+1)*iRowWidth*2 + rx*8 + 2;
	unsigned char *ob = smem + (BLOCKSIZEX + 4)*(BLOCKSIZEY + 4) + ry*oRowWidth*2 + rx*24;
	
	short regs[17];
	short gr1, gr2, gr3;

	regs[0] = ib[-iRowWidth2];
	regs[1] = ib[-iRowWidth-1];
	regs[2] = ib[-iRowWidth];
	regs[3] = ib[-iRowWidth+1];
	regs[4] = ib[-2];
	regs[5] = ib[-1];
	regs[6] = ib[0];
	regs[7] = ib[1];
	regs[8] = ib[2];
	regs[9] = ib[iRowWidth-1];
	regs[10] = ib[iRowWidth];
	regs[11] = ib[iRowWidth+1];
	regs[12] = ib[iRowWidth2];
	//r_0
	ob[0] = regs[6];
	gr1 = regs[0] + regs[4] + regs[8] + regs[12];
	
	ob[1] = clamp((4*regs[6] + 2*(regs[2] + regs[5] + regs[7] + regs[10]) - gr1)>>3);
	ob[2] = clamp((12*regs[6] + 4*(regs[1] + regs[3] + regs[9] + regs[11]) - gr1*3)>>4);
	
	// g2_0
	regs[0] = ib[iRowWidth - 2];
	regs[15] = ib[iRowWidth + 2];
	regs[1] = ib[iRowWidth2 - 1];
	regs[16] = ib[iRowWidth2 + 1];
	regs[4] = ib[iRowWidth3];
	
	gr1 = 10*regs[10] - 2*(regs[5] + regs[7] + regs[1] + regs[16]);
	gr2 = regs[0] + regs[15];
	gr3 = regs[2] + regs[4];
	ob[oRowWidth] = clamp((gr1 + 8*(regs[6]+regs[12]) - 2*gr3 + gr2)>>4);
	ob[oRowWidth+1] = regs[10];
	ob[oRowWidth+2] = clamp((gr1 + 8*(regs[9]+regs[11]) + gr3 - 2*gr2)>>4);

	//g1_0
	regs[0] = ib[-iRowWidth2 + 1];
	regs[1] = ib[-iRowWidth + 2];
	regs[4] = ib[3];

	gr1 = 10*regs[7] - 2*(regs[2] + regs[1] + regs[10] + regs[15]);
	gr2 = regs[5] + regs[4];
	gr3 = regs[0] + regs[16];

	ob[3] = clamp((gr1 + 8*(regs[6]+regs[8]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[7];
	ob[5] = clamp((gr1 + 8*(regs[3]+regs[11]) - 2*gr3 + gr2)>>4);

	//b_0
	regs[0] = ib[iRowWidth3+1];
	regs[5] = ib[iRowWidth2 + 2];
	regs[2] = ib[iRowWidth + 3];
	gr1 = regs[3] + regs[9] + regs[2] + regs[0];

	ob[oRowWidth+3] = clamp((12*regs[11] + 4*(regs[6] + regs[8] + regs[12] + regs[5]) - gr1*3)>>4);
	ob[oRowWidth+4] = clamp((4*regs[11] + 2*(regs[7] + regs[10] + regs[16] + regs[15]) - gr1)>>3);
	ob[oRowWidth+5] = regs[11];

	ib +=2; ob +=6;
	//r_1
	regs[0] = ib[-iRowWidth2];
	regs[9] = ib[-iRowWidth+1];
	regs[12] = ib[2];
	ob[0] = regs[8];
	gr1 = regs[0] + regs[6] + regs[5] + regs[12];
	ob[1] = clamp((4*regs[8] + 2*(regs[1] + regs[4] + regs[7] + regs[15]) - gr1)>>3);
	ob[2] = clamp((12*regs[8] + 4*(regs[2] + regs[3] + regs[9] + regs[11]) - gr1*3)>>4);

	// g2_1
	regs[3] = ib[iRowWidth + 2];
	regs[6] = ib[iRowWidth2 + 1];
	regs[0] = ib[iRowWidth3];
	
	gr1 = 10*regs[15] - 2*(regs[4] + regs[7] + regs[6] + regs[16]);
	gr2 = regs[10] + regs[3];
	gr3 = regs[1] + regs[0];
	ob[oRowWidth] = clamp((gr1 + 8*(regs[8]+regs[5]) - 2*gr3 + gr2)>>4);
	ob[oRowWidth+1] = regs[15];
	ob[oRowWidth+2] = clamp((gr1 + 8*(regs[2]+regs[11]) + gr3 - 2*gr2)>>4);

	//g1_1
	regs[0] = ib[-iRowWidth2 + 1];
	regs[10] = ib[-iRowWidth + 2];
	regs[16] = ib[3];

	gr1 = 10*regs[4] - 2*(regs[3] + regs[1] + regs[10] + regs[15]);
	gr2 = regs[7] + regs[16];
	gr3 = regs[0] + regs[6];

	ob[3] = clamp((gr1 + 8*(regs[12]+regs[8]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[4];
	ob[5] = clamp((gr1 + 8*(regs[2]+regs[9]) - 2*gr3 + gr2)>>4);

	//b_1
	regs[0] = ib[iRowWidth3+1];
	regs[7] = ib[iRowWidth2 + 2];
	regs[1] = ib[iRowWidth + 3];
	gr1 = regs[1] + regs[9] + regs[11] + regs[0];

	ob[oRowWidth+3] = clamp((12*regs[2] + 4*(regs[7] + regs[8] + regs[12] + regs[5]) - gr1*3)>>4);
	ob[oRowWidth+4] = clamp((4*regs[2] + 2*(regs[6] + regs[4] + regs[3] + regs[15]) - gr1)>>3);
	ob[oRowWidth+5] = regs[2];

	ib +=2; ob +=6;
	
	//r_2
	regs[0] = ib[-iRowWidth2];
	regs[5] = ib[-iRowWidth+1];
	regs[11] = ib[2];
	ob[0] = regs[12];
	gr1 = regs[0] + regs[8] + regs[7] + regs[11];
	ob[1] = clamp((4*regs[12] + 2*(regs[3] + regs[4] + regs[10] + regs[16]) - gr1)>>3);
	ob[2] = clamp((12*regs[12] + 4*(regs[2] + regs[1] + regs[9] + regs[5]) - gr1*3)>>4);

	// g2_2
	regs[9] = ib[iRowWidth + 2];
	regs[8] = ib[iRowWidth2 + 1];
	regs[0] = ib[iRowWidth3];
	
	gr1 = 10*regs[3] - 2*(regs[4] + regs[8] + regs[6] + regs[16]);
	gr2 = regs[15] + regs[9];
	gr3 = regs[10] + regs[0];
	ob[oRowWidth] = clamp((gr1 + 8*(regs[12]+regs[7]) - 2*gr3 + gr2)>>4);
	ob[oRowWidth+1] = regs[3];
	ob[oRowWidth+2] = clamp((gr1 + 8*(regs[2]+regs[1]) + gr3 - 2*gr2)>>4);

	//g1_2
	regs[0] = ib[-iRowWidth2 + 1];
	regs[6] = ib[-iRowWidth + 2];
	regs[15] = ib[3];

	gr1 = 10*regs[16] - 2*(regs[3] + regs[6] + regs[10] + regs[9]);
	gr2 = regs[4] + regs[15];
	gr3 = regs[0] + regs[8];

	ob[3] = clamp((gr1 + 8*(regs[12]+regs[11]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[16];
	ob[5] = clamp((gr1 + 8*(regs[5]+regs[1]) - 2*gr3 + gr2)>>4);

	//b_2
	regs[0] = ib[iRowWidth3+1];
	regs[4] = ib[iRowWidth2 + 2];
	regs[10] = ib[iRowWidth + 3];
	gr1 = regs[5] + regs[2] + regs[10] + regs[0];

	ob[oRowWidth+3] = clamp((12*regs[1] + 4*(regs[7] + regs[11] + regs[12] + regs[4]) - gr1*3)>>4);
	ob[oRowWidth+4] = clamp((4*regs[1] + 2*(regs[8] + regs[9] + regs[3] + regs[16]) - gr1)>>3);
	ob[oRowWidth+5] = regs[1];

	ib +=2; ob +=6;

	//r_3
	regs[0] = ib[-iRowWidth2];
	regs[7] = ib[-iRowWidth+1];
	regs[2] = ib[2];
	ob[0] = regs[11];
	gr1 = regs[0] + regs[2] + regs[4] + regs[12];
	ob[1] = clamp((4*regs[11] + 2*(regs[6] + regs[9] + regs[15] + regs[16]) - gr1)>>3);
	ob[2] = clamp((12*regs[11] + 4*(regs[7] + regs[1] + regs[10] + regs[5]) - gr1*3)>>4);

	// g2_3
	regs[12] = ib[iRowWidth + 2];
	regs[5] = ib[iRowWidth2 + 1];
	regs[0] = ib[iRowWidth3];
	
	gr1 = 10*regs[9] - 2*(regs[5] + regs[8] + regs[15] + regs[16]);
	gr2 = regs[12] + regs[3];
	gr3 = regs[6] + regs[0];
	ob[oRowWidth] = clamp((gr1 + 8*(regs[11]+regs[4]) - 2*gr3 + gr2)>>4);
	ob[oRowWidth+1] = regs[9];
	ob[oRowWidth+2] = clamp((gr1 + 8*(regs[10]+regs[1]) + gr3 - 2*gr2)>>4);

	//g1_2
	regs[0] = ib[-iRowWidth2 + 1];
	regs[8] = ib[-iRowWidth + 2];
	regs[3] = ib[3];

	gr1 = 10*regs[15] - 2*(regs[8] + regs[6] + regs[12] + regs[9]);
	gr2 = regs[3] + regs[16];
	gr3 = regs[0] + regs[5];

	ob[3] = clamp((gr1 + 8*(regs[2]+regs[11]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[15];
	ob[5] = clamp((gr1 + 8*(regs[7]+regs[10]) - 2*gr3 + gr2)>>4);

	//b_3
	regs[0] = ib[iRowWidth3+1];
	regs[6] = ib[iRowWidth2 + 2];
	regs[16] = ib[iRowWidth + 3];
	gr1 = regs[7] + regs[1] + regs[16] + regs[0];

	ob[oRowWidth+3] = clamp((12*regs[10] + 4*(regs[2] + regs[11] + regs[6] + regs[4]) - gr1*3)>>4);
	ob[oRowWidth+4] = clamp((4*regs[10] + 2*(regs[5] + regs[9] + regs[12] + regs[15]) - gr1)>>3);
	ob[oRowWidth+5] = regs[10];

}

template <size_t BLOCKSIZEX, size_t BLOCKSIZEY> __device__ void Malvar_hv (unsigned char* smem, size_t inputImgW, size_t inputImgH )
{
	const int threadId = threadIdx.x & 31;
	const int rx = threadId & 15;
	const int ry = threadIdx.x>>4;
	const int iRowWidth = BLOCKSIZEX + 4;
	const int iRowWidth2 = iRowWidth*2;
	const int iRowWidth3 = iRowWidth*3;
	const int oRowWidth = BLOCKSIZEX * 3;

	unsigned char *ib = smem + (ry+1)*iRowWidth*2 + rx*8 + 2;
	unsigned char *ob = smem + (BLOCKSIZEX + 4)*(BLOCKSIZEY + 4) + ry*oRowWidth*2 + rx*24;

	size_t RealBlockWidth = inputImgW - blockIdx.x*BLOCKSIZEX - 4;
    const size_t RealBlockHeight = inputImgH - blockIdx.y*BLOCKSIZEY - 4;
	//RealBlockWidth = (RealBlockWidth>outputBlockW)?outputBlockW:RealBlockWidth;

	short regs[17];
	short gr1, gr2, gr3;

	if (rx*8 >= RealBlockWidth || ry*2>=RealBlockHeight)
		return;

	regs[0] = ib[-iRowWidth2];
	regs[1] = ib[-iRowWidth-1];
	regs[2] = ib[-iRowWidth];
	regs[3] = ib[-iRowWidth+1];
	regs[4] = ib[-2];
	regs[5] = ib[-1];
	regs[6] = ib[0];
	regs[7] = ib[1];
	regs[8] = ib[2];
	regs[9] = ib[iRowWidth-1];
	regs[10] = ib[iRowWidth];
	regs[11] = ib[iRowWidth+1];
	regs[12] = ib[iRowWidth2];
	//r_0
	ob[0] = regs[6];
	gr1 = regs[0] + regs[4] + regs[8] + regs[12];
	ob[1] = clamp((4*regs[6] + 2*(regs[2] + regs[5] + regs[7] + regs[10]) - gr1)>>3);
	ob[2] = clamp((12*regs[6] + 4*(regs[1] + regs[3] + regs[9] + regs[11]) - gr1*3)>>4);

	// g2_0
	regs[15] = ib[iRowWidth + 2];
	regs[16] = ib[iRowWidth2 + 1];
	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth - 2];
		regs[1] = ib[iRowWidth2 - 1];
		regs[4] = ib[iRowWidth3];
	
		gr1 = 10*regs[10] - 2*(regs[5] + regs[7] + regs[1] + regs[16]);
		gr2 = regs[0] + regs[15];
		gr3 = regs[2] + regs[4];
		ob[oRowWidth] = clamp((gr1 + 8*(regs[6]+regs[12]) - 2*gr3 + gr2)>>4);
		ob[oRowWidth+1] = regs[10];
		ob[oRowWidth+2] = clamp((gr1 + 8*(regs[9]+regs[11]) + gr3 - 2*gr2)>>4);
	}

	if (rx*8 + 1 >= RealBlockWidth)
		return;
	//g1_0
	regs[0] = ib[-iRowWidth2 + 1];
	regs[1] = ib[-iRowWidth + 2];
	regs[4] = ib[3];

	gr1 = 10*regs[7] - 2*(regs[2] + regs[1] + regs[10] + regs[15]);
	gr2 = regs[5] + regs[4];
	gr3 = regs[0] + regs[16];

	ob[3] = clamp((gr1 + 8*(regs[6]+regs[8]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[7];
	ob[5] = clamp((gr1 + 8*(regs[3]+regs[11]) - 2*gr3 + gr2)>>4);

	//b_0
	regs[5] = ib[iRowWidth2 + 2];
	regs[2] = ib[iRowWidth + 3];
	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth3+1];
		gr1 = regs[3] + regs[9] + regs[2] + regs[0];

		ob[oRowWidth+3] = clamp((12*regs[11] + 4*(regs[6] + regs[8] + regs[12] + regs[5]) - gr1*3)>>4);
		ob[oRowWidth+4] = clamp((4*regs[11] + 2*(regs[7] + regs[10] + regs[16] + regs[15]) - gr1)>>3);
		ob[oRowWidth+5] = regs[11];
	}

	if (rx*8 + 2 >= RealBlockWidth)
		return;

	ib +=2; ob +=6;
	//r_1

	regs[0] = ib[-iRowWidth2];
	regs[9] = ib[-iRowWidth+1];
	regs[12] = ib[2];
	ob[0] = regs[8];
	gr1 = regs[0] + regs[6] + regs[5] + regs[12];
	ob[1] = clamp((4*regs[8] + 2*(regs[1] + regs[4] + regs[7] + regs[15]) - gr1)>>3);
	ob[2] = clamp((12*regs[8] + 4*(regs[2] + regs[3] + regs[9] + regs[11]) - gr1*3)>>4);

	// g2_1
	regs[3] = ib[iRowWidth + 2];
	regs[6] = ib[iRowWidth2 + 1];
	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth3];
	
		gr1 = 10*regs[15] - 2*(regs[4] + regs[7] + regs[6] + regs[16]);
		gr2 = regs[10] + regs[3];
		gr3 = regs[1] + regs[0];
		ob[oRowWidth] = clamp((gr1 + 8*(regs[8]+regs[5]) - 2*gr3 + gr2)>>4);
		ob[oRowWidth+1] = regs[15];
		ob[oRowWidth+2] = clamp((gr1 + 8*(regs[2]+regs[11]) + gr3 - 2*gr2)>>4);
	}

	if (rx*8 + 3 >= RealBlockWidth)
		return;
	//g1_1
	regs[0] = ib[-iRowWidth2 + 1];
	regs[10] = ib[-iRowWidth + 2];
	regs[16] = ib[3];

	gr1 = 10*regs[4] - 2*(regs[3] + regs[1] + regs[10] + regs[15]);
	gr2 = regs[7] + regs[16];
	gr3 = regs[0] + regs[6];

	ob[3] = clamp((gr1 + 8*(regs[12]+regs[8]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[4];
	ob[5] = clamp((gr1 + 8*(regs[2]+regs[9]) - 2*gr3 + gr2)>>4);

	//b_1
	regs[7] = ib[iRowWidth2 + 2];
	regs[1] = ib[iRowWidth + 3];

	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth3+1];
		gr1 = regs[1] + regs[9] + regs[11] + regs[0];

		ob[oRowWidth+3] = clamp((12*regs[2] + 4*(regs[7] + regs[8] + regs[12] + regs[5]) - gr1*3)>>4);
		ob[oRowWidth+4] = clamp((4*regs[2] + 2*(regs[6] + regs[4] + regs[3] + regs[15]) - gr1)>>3);
		ob[oRowWidth+5] = regs[2];
	}

	if (rx*8 + 4 >= RealBlockWidth)
		return;
	ib +=2; ob +=6;
	
	//r_2
	regs[0] = ib[-iRowWidth2];
	regs[5] = ib[-iRowWidth+1];
	regs[11] = ib[2];
	ob[0] = regs[12];
	gr1 = regs[0] + regs[8] + regs[7] + regs[11];
	ob[1] = clamp((4*regs[12] + 2*(regs[3] + regs[4] + regs[10] + regs[16]) - gr1)>>3);
	ob[2] = clamp((12*regs[12] + 4*(regs[2] + regs[1] + regs[9] + regs[5]) - gr1*3)>>4);

	// g2_2
	regs[9] = ib[iRowWidth + 2];
	regs[8] = ib[iRowWidth2 + 1];
	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth3];
	
		gr1 = 10*regs[3] - 2*(regs[4] + regs[8] + regs[6] + regs[16]);
		gr2 = regs[15] + regs[9];
		gr3 = regs[10] + regs[0];
		ob[oRowWidth] = clamp((gr1 + 8*(regs[12]+regs[7]) - 2*gr3 + gr2)>>4);
		ob[oRowWidth+1] = regs[3];
		ob[oRowWidth+2] = clamp((gr1 + 8*(regs[2]+regs[1]) + gr3 - 2*gr2)>>4);
	}

	if (rx*8 + 5 >= RealBlockWidth)
		return;
	//g1_2
	regs[0] = ib[-iRowWidth2 + 1];
	regs[6] = ib[-iRowWidth + 2];
	regs[15] = ib[3];

	gr1 = 10*regs[16] - 2*(regs[3] + regs[6] + regs[10] + regs[9]);
	gr2 = regs[4] + regs[15];
	gr3 = regs[0] + regs[8];

	ob[3] = clamp((gr1 + 8*(regs[12]+regs[11]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[16];
	ob[5] = clamp((gr1 + 8*(regs[5]+regs[1]) - 2*gr3 + gr2)>>4);

	//b_2
	regs[4] = ib[iRowWidth2 + 2];
	regs[10] = ib[iRowWidth + 3];
	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth3+1];
		gr1 = regs[5] + regs[2] + regs[10] + regs[0];

		ob[oRowWidth+3] = clamp((12*regs[1] + 4*(regs[7] + regs[11] + regs[12] + regs[4]) - gr1*3)>>4);
		ob[oRowWidth+4] = clamp((4*regs[1] + 2*(regs[8] + regs[9] + regs[3] + regs[16]) - gr1)>>3);
		ob[oRowWidth+5] = regs[1];
	}

	if (rx*8 + 6 >= RealBlockWidth)
		return;
	ib +=2; ob +=6;

	//r_3
	regs[0] = ib[-iRowWidth2];
	regs[7] = ib[-iRowWidth+1];
	regs[2] = ib[2];
	ob[0] = regs[11];
	gr1 = regs[0] + regs[2] + regs[4] + regs[12];
	ob[1] = clamp((4*regs[11] + 2*(regs[6] + regs[9] + regs[15] + regs[16]) - gr1)>>3);
	ob[2] = clamp((12*regs[11] + 4*(regs[7] + regs[1] + regs[10] + regs[5]) - gr1*3)>>4);

	// g2_3
	regs[12] = ib[iRowWidth + 2];
	regs[5] = ib[iRowWidth2 + 1];
	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth3];
	
		gr1 = 10*regs[9] - 2*(regs[5] + regs[8] + regs[15] + regs[16]);
		gr2 = regs[12] + regs[3];
		gr3 = regs[6] + regs[0];
		ob[oRowWidth] = clamp((gr1 + 8*(regs[11]+regs[4]) - 2*gr3 + gr2)>>4);
		ob[oRowWidth+1] = regs[9];
		ob[oRowWidth+2] = clamp((gr1 + 8*(regs[10]+regs[1]) + gr3 - 2*gr2)>>4);
	}

	if (rx*8 + 7 >= RealBlockWidth)
		return;
	//g1_2
	regs[0] = ib[-iRowWidth2 + 1];
	regs[8] = ib[-iRowWidth + 2];
	regs[3] = ib[3];

	gr1 = 10*regs[15] - 2*(regs[8] + regs[6] + regs[12] + regs[9]);
	gr2 = regs[3] + regs[16];
	gr3 = regs[0] + regs[5];

	ob[3] = clamp((gr1 + 8*(regs[2]+regs[11]) + gr3 - 2*gr2)>>4);
	ob[4] = regs[15];
	ob[5] = clamp((gr1 + 8*(regs[7]+regs[10]) - 2*gr3 + gr2)>>4);

	//b_3
	regs[6] = ib[iRowWidth2 + 2];
	regs[16] = ib[iRowWidth + 3];
	if (ry*2+1<RealBlockHeight)
	{
		regs[0] = ib[iRowWidth3+1];
		gr1 = regs[7] + regs[1] + regs[16] + regs[0];

		ob[oRowWidth+3] = clamp((12*regs[10] + 4*(regs[2] + regs[11] + regs[6] + regs[4]) - gr1*3)>>4);
		ob[oRowWidth+4] = clamp((4*regs[10] + 2*(regs[5] + regs[9] + regs[12] + regs[15]) - gr1)>>3);
		ob[oRowWidth+5] = regs[10];
	}
}

template <size_t BLOCKSIZEX, size_t BLOCKSIZEY> __global__ void malvarKernel(unsigned int* input, size_t ibrw, unsigned int* output, size_t obrw, size_t inputImgW, size_t inputImgH) {

    __shared__ unsigned char smem [(BLOCKSIZEX+4)*(BLOCKSIZEY+4) + BLOCKSIZEX*BLOCKSIZEY*3];  

    bool fastPath =  blockIdx.x < gridDim.x-1 && blockIdx.y < gridDim.y - 1;
	if (fastPath) 
        LoadGlobalToSM_g<BLOCKSIZEX, BLOCKSIZEY>(input, ibrw, smem);
     else
        LoadGlobalToSM_hv<BLOCKSIZEX, BLOCKSIZEY>(input, ibrw, smem, inputImgW, inputImgH);

     __syncthreads();
     if (fastPath) 
	 {
        Malvar_g<BLOCKSIZEX, BLOCKSIZEY> (smem);
		WriteSMtoGlobal_g<BLOCKSIZEX, BLOCKSIZEY>(output, obrw, smem);
	 }
     else
	 {
        Malvar_hv<BLOCKSIZEX, BLOCKSIZEY> (smem, inputImgW, inputImgH);
		WriteSMtoGlobal_hv<BLOCKSIZEX, BLOCKSIZEY>(output, obrw, smem, inputImgW, inputImgH);
	 }
}

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

int malvarGPUCall (Image& inputImg, unsigned int* inputBufCpu, size_t inputBufCpuWidth, int deviceId, unsigned char *_outputBufGpu)
{
	const size_t BLOCKX = 128;
    const size_t BLOCKY = 16;

	unsigned int *inputBufGpu;
    unsigned char *outputBufCpu;
	unsigned char *outputBufGpu;
	
    dim3 dimBlock(128);
    dim3 dimGrid((inputImg.w-4 + BLOCKX - 1) / BLOCKX, (inputImg.h - 4 + BLOCKY - 1) /BLOCKY);

    size_t outputBufWidth = inputBufCpuWidth;
    size_t outputBufHeight = inputImg.h;

	int outputBufSize = sizeof(unsigned char)*3*outputBufWidth*outputBufHeight;

	outputBufCpu = (unsigned char *)malloc(outputBufSize);
	cudaMalloc(&inputBufGpu,inputBufCpuWidth*sizeof(unsigned char)*inputImg.h);
	cudaMalloc(&outputBufGpu,outputBufSize);
		
	cudaMemcpy(inputBufGpu,inputBufCpu,inputBufCpuWidth*sizeof(unsigned char)*inputImg.h,cudaMemcpyHostToDevice);
	cudaMemset(outputBufGpu, 0, outputBufSize);
	
	memset( outputBufCpu, 0, outputBufSize );

	malvarKernel<BLOCKX, 16><<<dimGrid, dimBlock>>>(inputBufGpu, inputBufCpuWidth, (unsigned int*)outputBufGpu, 
		outputBufWidth, (size_t)inputImg.w, (size_t)inputImg.h  );
	cudaDeviceSynchronize();

//	cudaMemcpy(outputBufCpu,outputBufGpu,outputBufSize,cudaMemcpyDeviceToHost);
	cudaMemcpy(_outputBufGpu,outputBufGpu,outputBufSize,cudaMemcpyDeviceToDevice);
//	__savePPM("Output.ppm", outputBufCpu, outputBufWidth, outputBufHeight, 3);

	cudaFree(inputBufGpu);
	cudaFree(outputBufGpu);
	free(outputBufCpu);
	return 0;
}

void StartIt(Image inputImgCPU, unsigned char *outputBufGpu)
{
    size_t inputBufCpuWidth = ((inputImgCPU.w+127)/128)*128;

    unsigned int* inputBufCpu;
	inputBufCpu = (unsigned int*)malloc(sizeof(unsigned char)*inputBufCpuWidth*inputImgCPU.h);
	
    
    for (int i=0;i<inputImgCPU.h;i++)
    {
        memcpy(inputBufCpu+i*inputBufCpuWidth/4, inputImgCPU.data+inputImgCPU.w*i,inputImgCPU.w);
        for (int j=inputImgCPU.w;j<inputBufCpuWidth;j++)
        {
            unsigned char *inputBufCpuB =(unsigned char *)inputBufCpu;
            *(inputBufCpuB+inputBufCpuWidth*i+j) = 0;
        }
    }

	malvarGPUCall (inputImgCPU, inputBufCpu, inputBufCpuWidth, 0, outputBufGpu);
}

void StartIt2(Image inputImgCPU, unsigned char *outputBufGpu){
	checkCudaErrors(cudaMemcpy(outputBufGpu, inputImgCPU.data,
			inputImgCPU.w * inputImgCPU.h * 3, cudaMemcpyDeviceToDevice));
}
