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


#endif /* GLSLIDESHOW_H_ */
