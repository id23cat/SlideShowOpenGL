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

// global determines which filter to invoke
enum SobelDisplayMode {
	SOBELDISPLAY_IMAGE = 0, SOBELDISPLAY_SOBELTEX, SOBELDISPLAY_SOBELSHARED
};

extern enum SobelDisplayMode g_SobelDisplayMode;

extern "C" void sobelFilter(Pixel *odata, int iw, int ih,
		enum SobelDisplayMode mode, float fScale);
//extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);
extern "C" void initFilter(void);

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

	void CopyToGPU(Image inputImgCPU);
	void CopyFromGPU(Image *outputImgCPU);
	void SetDevice(int dev);
//	virtual Pixel* CallKernel(size_t *imWidth, size_t *imHeight)=0; // must return outputBufGpu;
	virtual Pixel* CallKernel(Pixel *outputBUFFER)=0;	// must return outputBufGpu;
														// outputBUFFER -- buf on GPU
	static void Reset();
};


class SobelKernel: public AbstractKernel{
//	cudaChannelFormatDesc desc;
//	cudaArray *array;
	enum SobelDisplayMode sobelDisplayMode;
	float imageScale;	// Image exposure
	cudaChannelFormatDesc desc;

public:
	SobelKernel();
	~SobelKernel();
	void SetPropeties(enum SobelDisplayMode mode, float fScale);
	void CopyToGPU(Image inputImgCPU, int pixSize=1);
	Pixel* CallKernel(Pixel *outputBUFFER);
};


class MalvarKernel: public AbstractKernel{
	const size_t BLOCKX;
	const size_t BLOCKY;
public:
	MalvarKernel();
	void CopyToGPU(Image inputImgCPU);
	Pixel* CallKernel(Pixel *outputBUFFER);

};

#endif /* INITCUDA_H_ */
