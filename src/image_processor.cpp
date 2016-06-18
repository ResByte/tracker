#include "image_processor.hpp"

/*	reesizes image to fixed size
	@params: input image and output image
*/
void ImageProcessor::resizeImg(cv::Mat& in, cv::Mat& out)
{
	cv::resize(in, out, cv::Size(_fixed_patch_size,_fixed_patch_size), 0,0, cv::INTER_CUBIC );
}

cv::Mat ImageProcessor::createGaussian()
{
	cv::Mat result = getGaussianKernel(51, 0.01, CV_32F);
	return result;
}

/* 	computes hog feature from the given image
	@params: greyscale image as input
	@params: output feature image

*/
void ImageProcessor::computeHoG(cv::Mat& img, cv::Mat& feature_image)
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

	cv::Mat hog_features;
	hog_features.create(desc.size(),1, CV_32FC1);

	// To display the results uncomment following
	for(auto i : desc )
	{
		hog_features.at<float>(i,0)=desc.at(i);
	}
	hog_features.copyTo(feature_image);
}

/*	computes rectangular ROI from given image
	@params: x,y coordinate of upper left corner of rectangle
	@params: width, height of rectangle from given coordinates
*/
void ImageProcessor::extractRect(cv::Mat& input, cv::Mat& output, int x, int y, int width, int height)
{
	// values are for object in the first image from ground truth file.
	cv::Mat subimage(input, cv::Rect(x,y,width,height));
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
	_curr_image  = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
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

/* 	convolves 2 input images in frequency domain
 	returns the result in frequency domain
 	@params: input image first as A
 	@params: input image second as B
 	@params: return output image
*/
cv::Mat ImageProcessor::convolveDFTSpectrum(cv::Mat& A, cv::Mat& B)
{
	//output.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols) +1, A.type());

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

	// copy the results to output
	//tempA(cv::Rect(0,0,output.cols, output.rows)).copyTo(output);
	return tempA;
}

/*	computes optimal correlation filter in frequency domain.
	It solves least square error minimzation to compute optimal filter
	for the given image desires output response is assumed gaussian with zero mean

*/
void ImageProcessor::getOptimalCorrelationFilter(cv::Mat input)
{
	// @TODO: check if input empty

	// convert input to greyscale
	cv::Mat grey = input;
	cv::cvtColor(grey, grey, CV_RGB2GRAY);

	// create gaussian kernel for desired response
	cv::Mat gauss_mat = getGaussianKernel(201, 100/16, CV_32F);

	// convlove previous roi with self and get frequency domain results
	cv::Mat s_hat =convolveDFTSpectrum(grey,grey);

	cv::Mat r_hat =convolveDFTSpectrum(gauss_mat,grey);

	// calculate current d_hat

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

/* Initialize first images and the roi based on groundtruth
	@params: path to file
*/
void ImageProcessor::initializeImages(std::string filename)
{
	_prev_image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::Mat window;
	extractRect(_prev_image, window, 243,165,110 ,115);
	cv::Mat resizedImg;
	resizeImg(window,resizedImg);
	//showImage(resizedImg);
	_prev_roi = resizedImg;

	// initialize position
	_p.x = 243;
	_p.y = 165;
	_p.w = 128;
	_p.h = 128;

	_prev_pos.x = 243 +115/2;
	_prev_pos.y = 165 + 110/2;
	// initialize scale
	_prev_scale.w = 110;
	_prev_scale.h = 115;

}

/*	Set current image from teh given filename
	@param: path to file
*/
void ImageProcessor::setCurrentImage(std::string filename)
{
	_curr_image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);


}


void ImageProcessor::initializeFilter(cv::Mat& y)
{
	y = cv::Mat::zeros(_p.w, _p.h, CV_32FC1);
	y.at<float>(_p.w/2, _p.h/2) = 1.0f;
	cv::GaussianBlur(y,y, cv::Size(_p.w/16+1,_p.w/16+1),_p.w/16,0);
	cv::normalize(y,y,cv::NORM_MINMAX);
}

/* 	computes inverse of matrix having imaginery components
	@params: input complex image image
	@params: output complex image
*/
void ImageProcessor::getComplexInverse(cv::Mat& in, cv::Mat& out)
{
	std::vector<cv::Mat> components;
	cv::split(in, components);
	cv::Mat real = components[0];
	cv::Mat imag = components[1];
	cv::Mat twice(in.rows*2,in.cols*2, CV_32FC1);
	real.copyTo(twice({0,0,in.cols,in.rows}));
	real.copyTo(twice({in.cols,in.rows,in.cols,in.rows}));
	imag.copyTo(twice({in.cols,0,in.cols,in.rows}));
	cv::Mat(-imag).copyTo(twice({0,in.rows,in.cols,in.rows}));

	cv::Mat twice_inv = twice.inv();
	twice_inv({0,0,in.cols,in.rows}).copyTo(real);
	twice_inv({in.cols, 0, in.cols, in.rows}).copyTo(imag);

	cv::Mat result(in.cols, in.rows, in.type());
	cv::merge(components, result);
	result.copyTo(out);
}

/*	computes model H for the image
	this is minimiser of the regularised objective
	@params: precessed input patch.
*/

void ImageProcessor::computeH(cv::Mat& img, ModelH& h_result)
{
	// 	calculate hog features for this image
	//	convert feature image from spatial domain to frequency domain
	//cv::Mat hog_feature_image;
	//computeHoG(_prev_roi, hog_feature_image);
	/* initialize with first frame */
	//	extract patch x to a fixed size
	cv::Mat resizedImg  = extractPatch(img);
	//showImage(resizedImg);
	// 	create Hanning window
	cv::Mat hann;
	cv::createHanningWindow(hann, cv::Size(_fixed_patch_size, _fixed_patch_size), CV_32F);
	hann.convertTo(hann, CV_32FC1, 1/255.0);

	// create desired output response y
	cv::Mat y = cv::Mat::zeros(_fixed_patch_size, _fixed_patch_size, CV_32FC1);
	y.at<float>(_fixed_patch_size/2, _fixed_patch_size/2) = 1.0f;
	cv::GaussianBlur(y,y, cv::Size(-1,-1),_fixed_patch_size/16,0);
	cv::normalize(y,y,cv::NORM_MINMAX);
	//showImage(y);
	// compute dft for desired output response y
	cv::Mat y_hat;
	cv::dft(y,y_hat, cv::DFT_COMPLEX_OUTPUT );

	// compute greyscale feature image
	cv::Mat phi; // feature image (currently grayscale image)
	cv::cvtColor(resizedImg, phi, CV_RGB2GRAY);
	cv::equalizeHist(phi, phi); // histogram equaizer for more contrast in image features
	phi.convertTo(phi, CV_32FC1,1/255.0 );

	//apply hann window before fourier transform
	phi = phi*hann;
	showImage(phi);
	// take fourier transform of feature image
	cv::Mat phi_hat;
	cv::dft(phi,phi_hat, cv::DFT_COMPLEX_OUTPUT );

	// multiply the spectrums to calculate r_hat(numerator) of model
	cv::Mat r_hat;
	cv::mulSpectrums(y_hat, phi_hat, r_hat,0,true);
	//showResponseImage(r_hat);
	// multiply the spectrums to calculate s_hat
	cv::Mat s_hat;
	cv::mulSpectrums(phi_hat,phi_hat, s_hat, 0, true);
	//showResponseImage(s_hat);




	// initialize model values
	cv::Mat prev_h_num = r_hat;
	cv::Mat d_hat = s_hat;
	cv::Mat prev_h_den;
	getComplexInverse(d_hat, prev_h_den);
	cv::Mat prev_h = prev_h_den * prev_h_num;
	showResponseImage(prev_h);

}

/*	Creates Training samples by randomly rotating image
	Used in MOSSE tracker
	@params: vector to store results and input sample image
*/
void ImageProcessor::createTrainingSample(std::vector<cv::Mat>& in, cv::Mat& sample)
{
	cv::Point center = cv::Point(sample.cols/2,sample.rows/2);
	double angle = 20.0;
	double scale = 1.0;

	Mat rot_mat( 2, 3, CV_32FC1 );
	rot_mat = getRotationMatrix2D(center, angle, scale);
	cv::Mat rotated;

	int i = 0;
	while( i < 8)
	{
		rot_mat = getRotationMatrix2D(center, angle + ((float)i)*angle, scale);
		cv::warpAffine(sample,rotated,rot_mat, sample.size(),cv::INTER_CUBIC );
		in.push_back(rotated);
		i++;
	}

}

// extracts patch of image for initialization
cv::Mat ImageProcessor::extractPatch(cv::Mat& in)
{
	// extract rectangle from the image with given dimension with no rotation
	int x1 = 243;
	int y1 = 165;
	int w = 110;
	int h = 115;
	int x = x1-(w/2);
	int y = y1 - (h/2);
	w = 2*w;
	h = 2*h;
	cv::Mat window;
	extractRect(in, window, x,y,w,h); // left most corner and width and height. taken heuristically
	// resize the patch to a given dimension
	cv::Mat resizedImg;
	resizeImg(window,resizedImg);
	return resizedImg;
}

// reads all image filenames in given directory
void ImageProcessor::readDir()
{

}

// shows response spectrum image
void ImageProcessor::showResponseImage(cv::Mat& img)
{
	cv::Mat res;
	cv::idft(img,res, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	res.convertTo(res, CV_32FC1, 1/255.0 );
	showImage(res);
}

// runs algorithm
void ImageProcessor::run()
{
	// load image
	_prev_image =  cv::imread("../vot15_car1/imgs/00000001.jpg", CV_LOAD_IMAGE_COLOR);
	_curr_image =  cv::imread("../vot15_car1/imgs/00000002.jpg", CV_LOAD_IMAGE_COLOR);

	ModelH h;
	computeH(_prev_image,h );

	//float prev_h_den = cv::trace(s_hat)[0];

	// initialize rectangle
	
}

/*
void ImageProcessor::run()
{
	initializeImages("../vot15_car1/imgs/00000001.jpg");
	setCurrentImage("../vot15_car1/imgs/00000002.jpg");

	std::vector<cv::Mat> training_samples;
	createTrainingSample(training_samples, _prev_roi);

	// for each sample compute model h
	std::vector<ModelH> training_models;
	//training
	for(auto i : training_samples)
	{
		showImage(i);
		ModelH h;
		computeH(i,h);
		training_models.push_back(h);
	}


	// Testing
	// extract patch around the same area as previous
	cv::Mat window;
	extractRect(_curr_image, window, _p.x,_p.y,_prev_scale.w ,_prev_scale.h);
	showImage(window);
	// resize the patch
	cv::Mat resizedImg;
	resizeImg(window,resizedImg);

	// compute h model for current patch
	ModelH curr_h;
	computeH(resizedImg,curr_h);

	// estimate max y  using learning rate parameter
	cv::Mat A_new = (1 - _learning_rate)*h.A + _learning_rate*curr_h.A;
	cv::Mat B_new = (1 - _learning_rate)*h.B + _learning_rate*curr_h.B;

	cv::Mat B_inv;
	getComplexInverse(B_new, B_inv);

	// resulting update on current image
	cv::Mat res;
	cv::mulSpectrums(B_inv, A_new, res,0, true);
	// compute inverse of the filter
	cv::dft(res,res, cv::DFT_INVERSE + cv::DFT_SCALE, A_new.rows);
	*/
	/*
	cv::normalize(res,res,cv::NORM_MINMAX);
	std::vector<cv::Mat> v;
	cv::split(res,v);
	std::cout<< res.size()<<std::endl;
	std::cout<< res.type()<<std::endl;
	//std::cout << "res = " << std::endl << " " <<res << std::endl << std::endl;
	showImage(v[0]);

}
*/
int main(int argc, char **argv)
{
	ImageProcessor processor;

	// processor.initializeImages("../vot15_car1/imgs/00000001.jpg");
	// processor.setCurrentImage("../vot15_car1/imgs/00000002.jpg");
	processor.run();

	return 0;
}
