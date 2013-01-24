/*
 * GLSlideShow.cpp
 *
 *  Created on: 29.12.2012
 *      Author: id23cat
 */
#include "GLSlideShow.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <string>

const char *SlideShowSample = WINDOW_NAME;

GLuint GLSlideShow::pbo_buffer = 0;
GLuint GLSlideShow::texid = 0;

GLSlideShow *glSH; // global pointer to one object

void gldisplay() {
	glSH->display();
}

void glkeyboard(unsigned char key, int x, int y) {
	glSH->keyboard(key, x, y);
}

void glreshape(int x, int y) {
	glSH->reshape(x, y);
}

GLSlideShow::GLSlideShow(int argc, char **argv) :
		frameCheckNumber(4) {
//	sSDKsample = WINDOW_NAME;

	initGL(&argc, argv);

	int dev = gpuGetMaxGflopsDeviceId();
	//
	cudaGLSetGLDevice(dev);

	wWidth = WIN_WIDTH; // Window width
	wHeight = WIN_HEIGHT; // Window height
	imWidth = 0; // Image width
	imHeight = 0; // Image height

	// Code to handle Auto verification
	fpsCount = 0; // FPS count for averaging
	fpsLimit = 8; // FPS limit for sampling
	frameCount = 0;
	g_TotalErrors = 0;
	timer = NULL;
	g_Index = 0;

	g_bQAReadback = false;

	pbo_buffer = 0;
	texid = 0;

//	pixels = NULL; // Image pixel data on the host
	imageScale = 1.f; // Image exposure
	//enum SobelDisplayMode g_SobelDisplayMode;

	pArgc = NULL;
	pArgv = NULL;

	pArgc = &argc;
	pArgv = argv;

	printf("%s Starting...\n\n", SlideShowSample);

	if (checkCmdLineFlag(argc, (const char **) argv, "help")) {
		printf("\nUsage: SlideShow <options>\n");
		printf("\t\t-mode=n (0=original, 1=texture, 2=smem + texture)\n");
		printf("\t\t-file=ref_orig.pgm (ref_tex.pgm, ref_shared.pgm)\n\n");
		exit(EXIT_SUCCESS);
	}

	//	    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
	//	    {
	//	        g_bQAReadback = true;
	//	        runAutoTest(argc, argv);
	//	    }
	//
	//	    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
	//	    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	//	    {
	//	        printf("   This SDK does not explicitly support -device=n when running with OpenGL.\n");
	//	        printf("   When specifying -device=n (n=0,1,2,....) the sample must not use OpenGL.\n");
	//	        printf("   See details below to run without OpenGL:\n\n");
	//	        printf(" > %s -device=n\n\n", argv[0]);
	//	        printf("exiting...\n");
	//	        exit(EXIT_SUCCESS);
	//	    }

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.

	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	glutDisplayFunc(gldisplay);
	glutKeyboardFunc(glkeyboard);
	glutReshapeFunc(glreshape);

}

GLSlideShow::~GLSlideShow() {
	cleanup();
}

void GLSlideShow::OpenFile(char* filename) {
	unsigned int w, h;
	size_t file_length = strlen(filename);
	printf("Opening file: %s\n", filename);
	Pixel *pixels;

	if (!strcmp(&filename[file_length - 3], "pgm")) {
		fprintf(stderr,"PGM");
		if (sdkLoadPGM<unsigned char>(filename, &pixels, &w, &h) != true) {
			fprintf(stderr,"Failed to load PGM image file: %s\n", filename);
			exit(EXIT_FAILURE);
		}

		g_Bpp = 1;
	} else if (!strcmp(&filename[file_length - 3], "ppm")) {
		fprintf(stderr,"PPM");

		if (sdkLoadPPM4(filename, &pixels, &w, &h) != true) {
			fprintf(stderr,"Failed to load PPM image file: %s\n", filename);
			exit(EXIT_FAILURE);
		}

		g_Bpp = 4;
	} else {
		fprintf(stderr,"Something is wrong");

		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	fprintf(stderr,"Load to device");
	LoadCPUtoGL(pixels, w, h, g_Bpp);
}

void GLSlideShow::LoadCPUtoGL(Pixel* cpumem, int width, int height, size_t pxsize) {
	GLint bsize;
	imWidth = width;
	imHeight = height;
	g_Bpp = pxsize;

	//    setupTexture(imWidth, imHeight, pixels, g_Bpp);
	devPointer = loadToDevice(imWidth, imHeight, cpumem, g_Bpp);

//	memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * imWidth * imHeight);

	if (!g_bQAReadback) {
		printf("Load init");
		// use OpenGL Path
		glGenBuffers(1, &pbo_buffer);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
		printf("Load start");
		glBufferData(GL_PIXEL_UNPACK_BUFFER,
				g_Bpp * sizeof(Pixel) * imWidth * imHeight, cpumem,
				GL_STREAM_DRAW);

		glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

		if ((GLuint) bsize != (g_Bpp * sizeof(Pixel) * imWidth * imHeight)) {
			printf("Buffer object (%d) has incorrect size (%d).\n",
					(unsigned) pbo_buffer, (unsigned) bsize);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
		printf("Load finish");

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// register this buffer object with CUDA
		checkCudaErrors(
				cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));

		glGenTextures(1, &texid);
		glBindTexture(GL_TEXTURE_2D, texid);
		glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp == 1) ? GL_LUMINANCE : GL_BGRA),
				imWidth, imHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
	}
	printf("Load ens");
}

void GLSlideShow::LoadCPUtoCUDA(Pixel* cpumem, int width, int height, size_t pxsize) {
	imWidth = width;
	imHeight = height;
	g_Bpp = pxsize;

	checkCudaErrors(
			cudaMalloc(&devPointer, imWidth*imHeight*g_Bpp));
	checkCudaErrors(
			cudaMemcpy(devPointer, cpumem, imWidth*imHeight*g_Bpp, cudaMemcpyHostToDevice));
}

void GLSlideShow::CUDAtoGL(Pixel* gpumem, int width, int height, size_t pxsize) {
	imWidth = width;
	imHeight = height;
	g_Bpp = pxsize;
}

void GLSlideShow::initGL(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(wWidth, wHeight);
	glutCreateWindow("CUDA Edge Detection");

	glewInit();

	if (!glewIsSupported(
			"GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, "  OpenGL version 1.5\n");
		fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
		exit(EXIT_FAILURE);
	}
}

void GLSlideShow::setImSize(int width, int height) {
	imWidth = width;
	imHeight = height;
}

void GLSlideShow::loadDefaultImage(char *loc_path){
	OpenFile("lena.pgm");
}

// This is the normal display path
void GLSlideShow::display(void) {
	sdkStartTimer(&timer);

	// Sobel operation
	Pixel *data = NULL;

	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(
			cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes, cuda_pbo_resource));
//    printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

//    sobelFilter(data, imWidth, imHeight, g_SobelDisplayMode, imageScale);
	checkCudaErrors(
			cudaMemcpy(data, devPointer, imWidth*imHeight, cudaMemcpyDeviceToDevice));

	// Run processing
	printf("Call kernel");
	fflush(stdout);
//	callKernel(data, imWidth, imHeight, g_Bpp);

//    printf("width = %d, height = %d\n", imWidth, imHeight);
//    checkCudaErrors(cudaMemset(data, 128, imWidth*imHeight*sizeof(Pixel)));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	glClear(GL_COLOR_BUFFER_BIT);

	glBindTexture(GL_TEXTURE_2D, texid);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight, GL_LUMINANCE,
			GL_UNSIGNED_BYTE, OFFSET(0));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glBegin(GL_QUADS);
	glVertex2f(0, 0);
	glTexCoord2f(0, 0);
	glVertex2f(0, 1);
	glTexCoord2f(1, 0);
	glVertex2f(1, 1);
	glTexCoord2f(1, 1);
	glVertex2f(1, 0);
	glTexCoord2f(0, 1);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glutSwapBuffers();

	sdkStopTimer(&timer);

	computeFPS();
}

void GLSlideShow::keyboard(unsigned char key, int /*x*/, int /*y*/) {
	char temp[256];

	switch (key) {
	case 27:
	case 'q':
	case 'Q':
		printf("Shutting down...\n");
		exit(EXIT_SUCCESS);
		break;

	case '-':
		imageScale -= 0.1f;
		printf("brightness = %4.2f\n", imageScale);
		break;

	case '=':
		imageScale += 0.1f;
		printf("brightness = %4.2f\n", imageScale);
		break;

	case 'i':
	case 'I':
//            g_SobelDisplayMode = SOBELDISPLAY_IMAGE;
		sprintf(temp, "CUDA Edge Detection ");
		glutSetWindowTitle(temp);
		break;

	case 's':
	case 'S':
//            g_SobelDisplayMode = SOBELDISPLAY_SOBELSHARED;
		sprintf(temp, "CUDA Edge Detection ");
		glutSetWindowTitle(temp);
		break;

	case 't':
	case 'T':
//            g_SobelDisplayMode = SOBELDISPLAY_SOBELTEX;
		sprintf(temp, "CUDA Edge Detection ");
		glutSetWindowTitle(temp);
		break;

	default:
		break;
	}
}

void GLSlideShow::reshape(int x, int y) {
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void GLSlideShow::cleanup(void) {
	cudaGraphicsUnregisterResource(cuda_pbo_resource);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &pbo_buffer);
	glDeleteTextures(1, &texid);
//    deleteTexture();
	deleteImage(devPointer);

	sdkDeleteTimer(&timer);
}

void GLSlideShow::computeFPS() {
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "CUDA Edge Detection : %3.1f fps", ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		sdkResetTimer(&timer);
	}
}
