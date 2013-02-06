/*
 * SlideShowOpenGL.cpp
 *
 *  Created on: 27.12.2012
 *      Author: id23cat
 */


#include "GLSlideShow.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#include "../common/inc/dirent.h"
#else //WIN32
#include </usr/include/dirent.h>
#endif


#include <vector>
#include <string>
#include <iostream>

int main(int argc, char **argv)
{
	struct dirent *dp;

	// enter existing path to directory below

	std::cout << "Alowed only .pgm files" << std::endl;

	std::string dir_path = ".";
	std::string extension = ".pgm";
//	std::string extension = ".ppm";
	if(argc > 1)
		dir_path = argv[1];

	dir_path.push_back('/');

	DIR *dir = opendir(dir_path.data());
	if(dir == NULL)
		return 0;
	std::vector<std::string> filesList;

	while ((dp = readdir(dir)) != NULL) {
		std::string file = dp->d_name;
		if(file.find(extension) != std::string::npos){
			file.insert(0, dir_path);
			filesList.push_back(file);
		}
	}
	closedir(dir);

	if(filesList.empty()){
		std::cerr << "No .pgm files foud";
		return 0;
	}

	for (std::vector<std::string>::iterator it = filesList.begin(); it != filesList.end(); ++it)
	    std::cout << ' ' << *it << std::endl;

	Start(argc, argv, filesList);
}
