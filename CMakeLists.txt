#The name of the project

if(UNIX) 
  set(CMAKE_C_COMPILER_VERSION 4.6.3) 
  SET (CMAKE_C_COMPILER "/usr/bin/gcc")
  message(STATUS "Compiler: GCC, version: " ${CMAKE_C_COMPILER_VERSION})
  set(CMAKE_CXX_COMPILER_VERSION 4.6.3) 
  SET (CMAKE_CXX_COMPILER "/usr/bin/g++")
  message(STATUS "Compiler: G++, version: " ${CMAKE_CXX_COMPILER_VERSION})
endif(UNIX) 

set (PROJECT SlideShowOpenGL)
PROJECT(${PROJECT})

# The CMake Minimum version that is required. The FindCUDA script
# is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Searching CUDA
FIND_PACKAGE(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Searching OpenGL
FIND_PACKAGE(OpenGL  REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})


set(COMMON_DIR ${CMAKE_SOURCE_DIR}/common)
include_directories(${COMMON_DIR}/inc)

if (WIN32)
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( LIB_PATH ${COMMON_DIR}/lib/win/x64/ )
    set( GLEW_NAME glew64 )
#     set(GENCODE -m64)

  else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( LIB_PATH ${COMMON_DIR}/lib/win/Win32/ )
    set( GLEW_NAME glew32 )
#     set(GENCODE -m32)
  endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

else (WIN32)
  set( GLEW_NAME GLEW )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( LIB_PATH ${COMMON_DIR}/lib/linux/x86_64/ )
#     set(GENCODE -m64)

  else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( LIB_PATH ${COMMON_DIR}/lib/linux/i686/ )    
#     set(GENCODE -m32)
  endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
endif (WIN32)

MESSAGE( "LIB_PATH = " ${LIB_PATH} )

#Searching GLEW
#FIND_PACKAGE(GLEW REQUIRED)
#include_directories(${GLEW_INCLUDE_DIRS})
#glew
FIND_PATH(GLEW_INCUDE_DIR NAMES GL/glew.h PATHS ${COMMON_DIR}/inc)
if (WIN32)
  FIND_LIBRARY(GLEW_LIBRARY NAMES ${GLEW_NAME} PATHS ${LIB_PATH})
else (WIN32)
  FIND_LIBRARY(GLEW_LIBRARY NAMES ${GLEW_NAME} PATHS ${LIB_PATH})
endif (WIN32)
MESSAGE( STATUS "GLEW_LIBRARY = " ${GLEW_LIBRARY} )
MESSAGE( STATUS "GLEW_INCUDE_DIR = " ${GLEW_INCUDE_DIR} )

#Searching GLUT
#FIND_PACKAGE(GLUT REQUIRED)
#include_directories(${GLUT_INCLUDE_DIRS})
#freeglut
FIND_PATH(FREEGLUT_INCLUDE_DIR NAMES GL/freeglut.h PATHS ${COMMON_DIR}/inc)
if (WIN32)
  FIND_LIBRARY(FREEGLUT_LIBRARY NAMES freeglut PATHS ${LIB_PATH})
else (WIN32)
  FIND_LIBRARY(FREEGLUT_LIBRARY NAMES glut PATHS ${LIB_PATH})
endif (WIN32)
MESSAGE( STATUS "FREEGLUT_LIBRARY = " ${FREEGLUT_LIBRARY} )
MESSAGE( STATUS "FREEGLUT_INCLUDE_DIR = " ${FREEGLUT_INCLUDE_DIR} )

option(CUDA_DEVICE_DEBUGGING
  "On to enable debugging device code."
  OFF
  )

if (CUDA_DEVICE_DEBUGGING)
  set(DEVICE_DEBUGGING -G0)
endif (CUDA_DEVICE_DEBUGGING)

SET( SHGL_HEADERS
 src/GLSlideShow.h
 src/initCUDA.h
)

SET( SHGL_SRC
  src/initCUDA.cu
  src/GLSlideShow.cpp
  src/SlideShowOpenGL.cpp
)

MESSAGE( "SHGL_SRC= " ${SHGL_SRC} )

# if (WIN32)
#   if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
#     set( LIB_PATH ${COMMON_DIR}/lib/x64/ )
#     set( GLEW_NAME glew64 )
# 
#   else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
#     set( LIB_PATH ${COMMON_DIR}/lib/win32/ )
#     set( GLEW_NAME glew32 )
#   endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
# 
# else (WIN32)
#   if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
#     set( LIB_PATH ${COMMON_DIR}/lib/linux/x86_64/ )
#     set( GLEW_NAME GLEW )
# 
#   else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
#     set( LIB_PATH ${COMMON_DIR}/lib/linux/i686/ )
#     set( GLEW_NAME GLEW )  
#   endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
# endif (WIN32)

# MESSAGE( "LIB_PATH = " ${LIB_PATH} )

# #glew
# FIND_PATH(GLEW_INCUDE_DIR NAMES GL/glew.h PATHS ${CUDA_SDK_ROOT_DIR}/common/inc)
# MESSAGE( "GLEW_INCUDE_DIR = " ${GLEW_INCUDE_DIR} )

# if (WIN32)
#   FIND_LIBRARY(GLEW_LIBRARY NAMES ${GLEW_NAME} PATHS ${LIB_PATH})
#   FIND_LIBRARY(GL_LIBRARY NAMES ${GLEW_NAME} PATHS ${LIB_PATH})
# else (WIN32)
#   FIND_LIBRARY(GLEW_LIBRARY NAMES ${GL_NAME} PATHS ${LIB_PATH})
#   FIND_LIBRARY(GL_LIBRARY NAMES ${GL_NAME} PATHS ${LIB_PATH})
# endif (WIN32)

# MESSAGE ("GLEW_LIBRARY = " ${GLEW_LIBRARY} )
# MESSAGE ("OPENGL_LIBRARY = " ${OPENGL_LIBRARY} )
# MESSAGE ("GLUT_LIBRARY = " ${GLUT_LIBRARY} )

if(MSVC)
  message(STATUS "Compiler: MSVC, version: " ${MSVC_VERSION})
endif(MSVC)
if(BORLAND) 
  message(STATUS "Compiler: BCC") 
endif(BORLAND) 


message(STATUS "CMake generates " ${CMAKE_GENERATOR})

list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)

set(GENCODE ${GENCODE} -gencode arch=compute_10,code=sm_10)
# set(GENCODE ${GENCODE} -gencode arch=compute_20,code=sm_20 )
#set(GENCODE ${GENCODE} -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)

CUDA_ADD_EXECUTABLE( ${PROJECT} ${SHGL_SRC}
		     OPTIONS ${GENCODE} ${DEVICE_DEBUGGING})

target_link_libraries(${PROJECT} ${GLEW_LIBRARY} ${FREEGLUT_LIBRARY} ${OPENGL_LIBRARIES})
#${OPENGL_LIBRARIES}