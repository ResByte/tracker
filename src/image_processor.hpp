#ifndef _IMAGE_PROCESSOR_HPP 
#define _IMAGE_PROCESSOR_HPP
/*
 * image_processor.hpp
 * 
 * Copyright 2016 abhinav
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;

class ImageProcessor
{
public:
	ImageProcessor(){}	
	
	// reads image from file name 
	// displays until the window is closed
	void readImage(std::string filename);	
	
	// displays image untill shutdown
	void showImage(cv::Mat im);
	//sets filename 
	void setFileName(std::string name);

	
	
	// compute hog features from given image
	void computeHoG(cv::Mat& img);

	// convolves two arrays in frequency domain using dft
	void convolveDFT(cv::Mat& A, cv::Mat& B, cv::Mat& output );

	// get correlation filter for the image and patch
	void correlationFilter(cv::Mat& img, cv::Mat& filter, cv::Mat& output); 

	// extract subimage from the given image
	void extractRect(cv::Mat& input, cv::Mat& output, int x, int y, int width, int height);



	std::string _filename;
	cv::Mat _image;
	cv::HOGDescriptor _hog;
};
#endif /* _IMAGE_PROCESSOR_HPP */