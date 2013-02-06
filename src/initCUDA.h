/*
 * initCUDA.h
 *
 *  Created on: 27.12.2012
 *      Author: id23cat
 */

#ifndef INITCUDA_H_
#define INITCUDA_H_

typedef unsigned char Pixel;
struct Image
{
	Pixel *data;
	unsigned int w;
	unsigned int h;
	Image(Pixel *d, int width, int height):
		data(d), w(width), h(height){};
};

struct ImageC3
{
	Pixel *data;
	unsigned int w;
	unsigned int h;
};

void StartIt(Image inputImgCPU, unsigned char *outputBufGpu);
void StartIt2(Image inputImgCPU, unsigned char *outputBufGpu);

/********************************************************************************************/
class AbstractKernel {
protected:
	dim3 dimBlock; // block dimension
	dim3 dimGrid; // grid dimension

	Pixel *inputBufGpu; // input buffer on GPU
	size_t inBufWidth;
	size_t inBufHeight;

	Pixel *outputBufGpu; // output buffer on GPU
	size_t outBufWidth;
	size_t outBufHeight;

	int deviceIdx;

public:
	AbstractKernel();
	virtual ~AbstractKernel();

	virtual void CopyToGPU(Image inputImgCPU);
	virtual void CopyFromGPU(Image *outputImgCPU);
	void SetDevice(int dev);
//	virtual Pixel* CallKernel(size_t *imWidth, size_t *imHeight)=0; // must return outputBufGpu;
	virtual Pixel* CallKernel(Pixel *outputBUFFER)=0;	// must return outputBufGpu;
														// outputBUFFER -- buf on GPU
	static void Reset();
};
#endif /* INITCUDA_H_ */
