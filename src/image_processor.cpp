#include "image_processor.hpp"


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

/*	computes rectangular ROI from given image
	@params: x,y coordinate of upper left corner of rectangle
	@params: width, height of rectangle from given coordinates
*/
void ImageProcessor::extractRect(cv::Mat& input, cv::Mat& output, int x, int y, int width, int height)
{
	// values are for object in the first image from ground truth file. 
	cv::Mat subimage(input, cv::Rect(x,y,width,height));
	showImage(subimage);
	subimage.copyTo(output);

}

/*	displays image as an output untill a key is pressed
	@params: image as input
*/
void ImageProcessor::showImage(cv::Mat im)
{
	cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display", im);
	cv::waitKey(0);	
}

/*	reads image from the set filename
	@params: filename with path
*/
void ImageProcessor::readImage(std::string filename)
{
	_image  = cv::imread(filename, CV_LOAD_IMAGE_COLOR);	
}

/*	set global filename of the input image
	@params: input image filename with path
*/
void ImageProcessor::setFileName(std::string name)
{
	_filename = name;
}


/* 	convolves 2 input images in frequency domain 
 	returns the result in spatial domain
 	@params: input image first as A
 	@params: input image second as B
 	@params: resulting output image 
*/
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

/* 	filters an image in the frquency domain
 	computes fourier transform of image and filter
 	multiply in frequency domain
 	outputs the result by taking inverse transform of product
	@params: input image as im
	@params: input filter
	@params: resulting output image

	@TODO: perform check before input
*/
void ImageProcessor::correlationFilter(cv::Mat& im, cv::Mat& filter, cv::Mat& output)
{
	// using convolve dft 
	convolveDFT(im, filter,output);
}


int main(int argc, char **argv)
{
	ImageProcessor processor;
	processor.setFileName(argv[1]);
	processor.readImage(argv[1]);
	cv::Mat sub;
	processor.extractRect(processor._image, sub, 243,165,110 ,115);
	//processor.computeHoG(processor._image);
	return 0;
}

