/*
 * GLSlideShow.h
 *
 *  Created on: 29.12.2012
 *      Author: id23cat
 */

#ifndef GLSLIDESHOW_H_
#define GLSLIDESHOW_H_

#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
//#include <string.h>

#include <vector>
#include <string>

// includes, project
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking



//
// Cuda example code that implements the Sobel edge detection
// filter. This code works for 8-bit monochrome images.
//
// Use the '-' and '=' keys to change the scale factor.
//
// Other keys:
// I: display image
// T: display Sobel edge detection (computed solely with texture)
// S: display Sobel edge detection (computed with texture and shared memory)

//void cleanup(void);
//void initializeData(char *file) ;

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY     10 //ms

extern const char *SlideShowSample;

static int wWidth   = 512; // Window width
static int wHeight  = 512; // Window height
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height

// Code to handle Auto verification
const int frameCheckNumber = 4;
extern int fpsCount;      // FPS count for averaging
extern int fpsLimit;      // FPS limit for sampling
extern unsigned int frameCount;
extern unsigned int g_TotalErrors;
extern StopWatchInterface *timer;
extern unsigned int g_Bpp;
extern unsigned int g_Index;

extern bool g_bQAReadback;

// Display Data
static GLuint pbo_buffer = 0;  // Front and back CA buffers
extern struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

static GLuint texid = 0;       // Texture for display
extern unsigned char *pixels;  // Image pixel data on the host
extern float imageScale;        // Image exposure


extern int *pArgc;
extern char **pArgv;

extern "C" void runAutoTest(int argc, char **argv);

#define OFFSET(i) ((char *)NULL + (i))
#define MAX(a,b) ((a > b) ? a : b)

void computeFPS();

// This is the normal display path
void display(void);
void display1(void);

void timerEvent(int value);
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void reshape(int x, int y);
void cleanup(void);
void initializeData(/*char *file*/);
void loadImage(/*char *loc_exec*/);
void initGL(int *argc, char **argv);
//void runAutoTest(int argc, char *argv[]);

void Start(int argc, char **argv, std::vector<std::string> fileList);

//#include <GL/glew.h>
//#if defined(__APPLE__) || defined(MACOSX)
//#include <GLUT/glut.h>
//#else
//#include <GL/freeglut.h>
//#endif
//
//// CUDA utilities and system includes
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
//
//// Includes
//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//#include <string>
//
//#include "initCUDA.h"
//
//// includes, project
//#include <helper_functions.h> // includes for SDK helper functions
//#include <helper_cuda.h>      // includes for cuda initialization and error checking
//typedef unsigned char Pixel;
//
//#define REFRESH_DELAY	10 //ms
//#define WINDOW_NAME		"SlideShow OpenGL"
//#define WIN_WIDTH		512 // window width at start
//#define WIN_HEIGHT		512 // window height at start
//#define OFFSET(i) ((char *)NULL + (i))
//#define MAX(a,b) ((a > b) ? a : b)
//
///*
// *
// */
//class GLSlideShow {
////	const char sSDKsample[] = WINDOW_NAME;
//
//	Pixel* devPointer;
//	int wWidth; // Window width
//	int wHeight; // Window height
//	int imWidth; // Image width
//	int imHeight; // Image height
//
//	// Code to handle Auto verification
//	const int frameCheckNumber;
//	int fpsCount; // FPS count for averaging
//	int fpsLimit; // FPS limit for sampling
//	unsigned int frameCount;
//	unsigned int g_TotalErrors;
//	StopWatchInterface *timer;
//	unsigned int g_Bpp;		// pixel size in bytes
//	unsigned int g_Index;
//
//	bool g_bQAReadback;
//
//	// Display Data
//	static GLuint pbo_buffer; // Front and back CA buffers
//	static GLuint texid; // Texture for display
//	struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
//
////	unsigned char *pixels; // Image pixel data on the host
//	float imageScale; // Image exposure
//	//enum SobelDisplayMode g_SobelDisplayMode;
//
//	int *pArgc;
//	char **pArgv;
//public:
//	GLSlideShow(int argc, char **argv);
//	virtual ~GLSlideShow();
//
//	void OpenFile(char *filename);
//	void LoadCPUtoGL(Pixel* cpumem, int width, int height, size_t pxsize=1);	// pxsize -- pixel size in bytes
//	void LoadCPUtoCUDA(Pixel* cpumem, int width, int height, size_t pxsize=1);
//	void CUDAtoGL(Pixel* gpumem, int width, int height, size_t pxsize=1);
//
//	// callback function that call kernel
//	Pixel* (*callKernel)(Pixel* data, int width, int height, size_t pxsize);
//	void Run(){
////		glutTimerFunc(REFRESH_DELAY, timerEvent,0);
//		glutMainLoop();
//
//		cudaDeviceReset();
//	}
//
//	void setImSize(int width, int height);
//
//private:
//	void loadDefaultImage(char *loc_path);
//
//	friend void gldisplay();
//	friend void glkeyboard(unsigned char key, int /*x*/, int /*y*/);
//	friend void glreshape(int x, int y);
//
//	void initGL(int *argc, char **argv);
//
//	void display(void);
//	void keyboard(unsigned char key, int /*x*/, int /*y*/);
//	void reshape(int x, int y);
//
//	void computeFPS();
//	void timerEvent(int value);
//	void cleanup(void);
//};


#endif /* GLSLIDESHOW_H_ */
