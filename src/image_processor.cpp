/*
 * image_processor.cpp
 * 
 * Copyright 2016 abhi <abhi@abhi-MacBookPro>
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
	void readImage();	
	
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
	void extractRect(cv::Mat& input, cv::Mat& output);



	std::string _filename;
	cv::Mat _image;
	cv::HOGDescriptor _hog;
};

/* 	computes hog feature from the given image
		@params: greyscale image as input
		@params: descriptor as vector float container
		@params: sliding wndow size 
		@params: training padding 
		@params: locations  
*/
void ImageProcessor::computeHoG(cv::Mat& img)
{
	cv::Mat im = img;
	cv::cvtColor(im, im, CV_RGB2GRAY);
	 // create hog descriptor
	std::vector<float> desc; // create container to store descriptor
	std::vector<cv::Point> locs; // store locations
	
	// computes desc from hog of window size 4,4 and padding of 0,0 
	_hog.compute(im, desc, cv::Size(4,4), cv::Size(0,0), locs);

	//detection
	_hog.detect(im, locs,0, cv::Size(4,4), cv::Size(0,0));

	// To display the results uncomment following 
	// for(auto i : desc )
	// {
	// 	std::cout<< i<<std::endl;
	// }
}

void ImageProcessor::extractRect(cv::Mat& input, cv::Mat& output)
{
	// values are for object in the first image from ground truth file. 
	cv::Mat subimage(input, cv::Rect(243,165,110 ,115 ));
	showImage(subimage);
	subimage.copyTo(output);

}

void ImageProcessor::showImage(cv::Mat im)
{
	cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display", im);
	cv::waitKey(0);	
}

void ImageProcessor::readImage()
{
	_image  = cv::imread(_filename, CV_LOAD_IMAGE_COLOR);	
}

void ImageProcessor::setFileName(std::string name)
{
	_filename = name;
}


// from opencv documentation
// convolves 2 input images in frequency domain 
// returns the result in spatial domain
void ImageProcessor::convolveDFT(cv::Mat& A, cv::Mat& B, cv::Mat& output)
{
	output.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols) +1, A.type());

	// calculates size of dft transform
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = cv::getOptimalDFTSize(A.rows + B.rows -1);

	// allocate a temporary buffer
	cv::Mat tempA(dftSize, A.type(),cv::Scalar::all(0));
	cv::Mat tempB(dftSize, B.type(),cv::Scalar::all(0));

	cv::Mat roiA(tempA, cv::Rect(0,0,A.cols, A.rows));
	A.copyTo(roiA);
	cv::Mat roiB(tempB, cv::Rect(0,0,B.cols, B.rows));
	B.copyTo(roiB);

	// transform
	cv::dft(tempA, tempA, 0, A.rows);
	cv::dft(tempB, tempB, 0, B.rows);

	// multiply the spectrum
	// flag is to set the first array's conjugate before multiplying
	// default true
	bool flag = true;
	cv::mulSpectrums(tempA, tempB, tempA, flag);

	// take inverse fourier transform
	cv::dft(tempA, tempA, cv::DFT_INVERSE + cv::DFT_SCALE, output.rows);

	// copy the results to output
	tempA(cv::Rect(0,0,output.cols, output.rows)).copyTo(output);

}

// filters an image in the frquency domain
// computes fourier transform of image and filter
// multiply in frequency domain
// outputs the result by taking inverse transform of product
void ImageProcessor::correlationFilter(cv::Mat& im, cv::Mat& filter, cv::Mat& output)
{
	// TODO: perform checks

	// using convolve dft 
	convolveDFT(im, filter,output);
}


int main(int argc, char **argv)
{
	ImageProcessor processor;
	processor.setFileName(argv[1]);
	processor.readImage();
	cv::Mat sub;
	processor.extractRect(processor._image, sub);
	//processor.computeHoG(processor._image);
	return 0;
}

