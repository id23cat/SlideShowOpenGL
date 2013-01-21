/*
 * initCUDA.h
 *
 *  Created on: 27.12.2012
 *      Author: id23cat
 */

#ifndef INITCUDA_H_
#define INITCUDA_H_

typedef unsigned char Pixel;

Pixel *loadToDevice(int width, int height, Pixel *pixels, int pixelsize);

void deleteImage(Pixel *devPtr);

#endif /* INITCUDA_H_ */
