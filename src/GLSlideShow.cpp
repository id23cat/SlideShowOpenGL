/*
 * GLSlideShow.cpp
 *
 *  Created on: 29.12.2012
 *      Author: id23cat
 */

#include "GLSlideShow.h"
#include "initCUDA.h"

AbstractKernel *processKernel;

const char *filterMode[] =
{
    "No Filtering",
    "Sobel Texture",
    "Sobel SMEM+Texture",
    NULL
};

const char *SlideShowSample = "Slideshow";

enum SobelDisplayMode g_SobelDisplayMode;

int fpsCount = 0;      // FPS count for averaging
int fpsLimit = 8;      // FPS limit for sampling

unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

StopWatchInterface *timer = NULL;
unsigned int g_Bpp;
unsigned int g_Index = 0;
bool g_bQAReadback = false;
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

unsigned char *pixels = NULL;  // Image pixel data on the host

int *pArgc   = NULL;
char **pArgv = NULL;
float imageScale = 1.f;        // Image exposure

std::vector<std::string> *files;
std::vector<std::string>::iterator fileit;

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "CUDA Edge Detection (%s): %3.1f fps", filterMode[g_SobelDisplayMode], ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        sdkResetTimer(&timer);
    }
}


// This is the normal display path
void display(void)
{
    sdkStartTimer(&timer);

    // Sobel operation
    Pixel *data = NULL;

    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

//    sobelFilter(data, imWidth, imHeight, g_SobelDisplayMode, imageScale);
    processKernel->CallKernel(data);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texid);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight,
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, OFFSET(0));
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

void display1(void)
{
    sdkStartTimer(&timer);

    // Sobel operation
    Pixel *data = NULL;

    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    sobelFilter(data, imWidth, imHeight, g_SobelDisplayMode, imageScale);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texid);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight,
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, OFFSET(0));
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

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    char temp[256];
    char* file;
    GLint bsize;

    switch (key)
    {
        case 27:
        case 'q':
        case 'Q':
            printf("Shutting down...\n");
            exit(EXIT_SUCCESS);
            break;

		case 'p':
		case 'P':
			if (fileit == files->begin())
				fileit = (files->end());
			--fileit;
			file = (char*) (*fileit).data();

			sprintf(temp, "CUDA Sideshow (%s)", file);
			glutSetWindowTitle(file);

			loadImage(file);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
			glBufferData(GL_PIXEL_UNPACK_BUFFER,
					g_Bpp * sizeof(Pixel) * imWidth * imHeight, pixels,
					GL_STREAM_DRAW);

			glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

			if ((GLuint) bsize != (g_Bpp * sizeof(Pixel) * imWidth * imHeight)) {
				printf("Buffer object (%d) has incorrect size (%d).\n",
						(unsigned) pbo_buffer, (unsigned) bsize);
				cudaDeviceReset();
				exit(EXIT_FAILURE);
			}
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			glBindTexture(GL_TEXTURE_2D, texid);
			glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp == 1) ? GL_LUMINANCE : GL_BGRA),
					imWidth, imHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
			glBindTexture(GL_TEXTURE_2D, 0);

			break;

        case 'n':
        case 'N':
        	if(++fileit == files->end())
        		fileit = files->begin();
        	file = (char*)(*fileit).data();
        	sprintf(temp, "CUDA Sideshow (%s)",
        			file);
        	glutSetWindowTitle(file);

        	loadImage(file);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
			glBufferData(GL_PIXEL_UNPACK_BUFFER,
					g_Bpp * sizeof(Pixel) * imWidth * imHeight, pixels,
					GL_STREAM_DRAW);

			glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

			if ((GLuint) bsize != (g_Bpp * sizeof(Pixel) * imWidth * imHeight)) {
				printf("Buffer object (%d) has incorrect size (%d).\n",
						(unsigned) pbo_buffer, (unsigned) bsize);
				cudaDeviceReset();
				exit(EXIT_FAILURE);
			}
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
			glBindTexture(GL_TEXTURE_2D, texid);
			glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp == 1) ? GL_LUMINANCE : GL_BGRA),
					imWidth, imHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
			glBindTexture(GL_TEXTURE_2D, 0);

        	break;

        default:
            break;
    }
//    processKernel->SetPropeties(g_SobelDisplayMode, imageScale);
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void cleanup(void)
{
    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_buffer);
    glDeleteTextures(1, &texid);
//    deleteTexture();
    delete processKernel;

    sdkDeleteTimer(&timer);
}

void initializeData(/*char *file*/)
{
	GLint bsize;

	loadImage((char*)(*fileit).data());

//    setupTexture(imWidth, imHeight, pixels, g_Bpp);
//    sobelKernel->CopyToGPU(Image(pixels, imWidth, imHeight), g_Bpp);

    memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * imWidth * imHeight);

	if (!g_bQAReadback) {
		// use OpenGL Path
		glGenBuffers(1, &pbo_buffer);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
		glBufferData(GL_PIXEL_UNPACK_BUFFER,
				g_Bpp * sizeof(Pixel) * imWidth * imHeight, pixels,
				GL_STREAM_DRAW);

		glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

		if ((GLuint) bsize != (g_Bpp * sizeof(Pixel) * imWidth * imHeight)) {
			printf("Buffer object (%d) has incorrect size (%d).\n",
					(unsigned) pbo_buffer, (unsigned) bsize);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

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
}

void loadImage(char *loc_file)
{
//	const char *file = (*files)[0].data();
	unsigned int w, h;
	size_t file_length = strlen(loc_file);
	printf("Loading image %s...\n", loc_file);

	g_Bpp = 1;
	if(pixels != NULL){
		free(pixels);
		pixels = NULL;
	}

	if (sdkLoadPGM<unsigned char>(loc_file, &pixels, &w, &h) != true) {
		printf("Failed to load PGM image file: %s\n", loc_file);
		exit(EXIT_FAILURE);
	}

	//    if (!strcmp(&file[file_length-3], "pgm"))
	//    {
	//        if (sdkLoadPGM<unsigned char>(file, &pixels, &w, &h) != true)
	//        {
	//            printf("Failed to load PGM image file: %s\n", file);
	//            exit(EXIT_FAILURE);
	//        }
	//
	//        g_Bpp = 1;
	//    }
	//    else if (!strcmp(&file[file_length-3], "ppm"))
	//    {
	//        if (sdkLoadPPM4(file, &pixels, &w, &h) != true)
	//        {
	//            printf("Failed to load PPM image file: %s\n", file);
	//            exit(EXIT_FAILURE);
	//        }
	//
	//        g_Bpp = 4;
	//    }
	//    else
	//    {
	//        cudaDeviceReset();
	//        exit(EXIT_FAILURE);
	//    }

	imWidth = (int) w;
	imHeight = (int) h;

	printf("ANew image %dx%d\n", imWidth, imHeight);
//    processKernel = new SobelKernel();
    processKernel->CopyToGPU(Image(pixels, imWidth, imHeight));

}


void initGL(int *argc, char **argv){
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("CUDA Edge Detection");

    glewInit();

    if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_FAILURE);
    }
}


void Start(int argc, char **argv, std::vector<std::string> fileList){
	pArgc = &argc;
	pArgv = argv;

	printf("%s Starting...\n\n", SlideShowSample);

//	if (checkCmdLineFlag(argc, (const char **) argv, "help")) {
//		printf("\nUsage: SobelFilter <options>\n");
//		printf("\t\t-mode=n (0=original, 1=texture, 2=smem + texture)\n");
//		printf("\t\t-file=ref_orig.pgm (ref_tex.pgm, ref_shared.pgm)\n\n");
//		exit(EXIT_SUCCESS);
//	}
//
//	if (checkCmdLineFlag(argc, (const char **) argv, "file")) {
//		g_bQAReadback = true;
//		runAutoTest(argc, argv);
//	}
//
//	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
//	if (checkCmdLineFlag(argc, (const char **) argv, "device")) {
//		printf(
//				"   This SDK does not explicitly support -device=n when running with OpenGL.\n");
//		printf(
//				"   When specifying -device=n (n=0,1,2,....) the sample must not use OpenGL.\n");
//		printf("   See details below to run without OpenGL:\n\n");
//		printf(" > %s -device=n\n\n", argv[0]);
//		printf("exiting...\n");
//		exit(EXIT_SUCCESS);
//	}

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL(&argc, argv);
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);

	files = &fileList;
	fileit = files->begin();

	processKernel = new SobelKernel();	// creat SobelKernel
	initializeData();

//	loadImage(argv[0]);

	// If code is not printing the USage, then we execute this path.
	printf("N: next image\n");
	printf("P: previous image\n");
	printf("Esc or Q: exit\n");
	fflush(stdout);
	atexit(cleanup);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutMainLoop();

	cudaDeviceReset();
}
