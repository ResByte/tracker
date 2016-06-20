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

	// // To display the results uncomment following
	// for(auto i : desc )
	// {
	// 	hog_features.at<float>(i,0)=desc.at(i);
	// }
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

	bool flag = true;
	cv::mulSpectrums(A, B, output, flag);
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

	/* initialize with first frame */
	//	extract patch x to a fixed size
	cv::Mat resizedImg;
	img.copyTo(resizedImg);

	// 	create Hanning window
	cv::Mat hann;
	cv::createHanningWindow(hann, cv::Size(_fixed_patch_size, _fixed_patch_size), CV_32F);
	hann.convertTo(hann, CV_32FC1, 1/255.0);

	// create desired output response y
	cv::Mat y = cv::Mat::zeros(_fixed_patch_size, _fixed_patch_size, CV_32FC1);
	y.at<float>(_fixed_patch_size/2, _fixed_patch_size/2) = 1.0f;
	cv::GaussianBlur(y,y, cv::Size(-1,-1),_fixed_patch_size/16,0);
	cv::normalize(y,y,cv::NORM_MINMAX);

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

	// take fourier transform of feature image
	cv::Mat phi_hat;
	cv::dft(phi,phi_hat, cv::DFT_COMPLEX_OUTPUT );

	// multiply the spectrums to calculate r_hat(numerator) of model
	cv::Mat r_hat;
	cv::mulSpectrums(y_hat, phi_hat, r_hat,0,true);

	// multiply the spectrums to calculate s_hat
	cv::Mat s_hat;
	cv::mulSpectrums(phi_hat,phi_hat, s_hat, 0, true);

	r_hat.copyTo(h_result.r);
	s_hat.copyTo(h_result.s);
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

// pre-processing input image 

void ImageProcessor::preprocessImg(cv::Mat& img)
{
	// assumes single channel image
	// pre-processing based on Mosse tracker
	cv::cvtColor(img, img, CV_RGB2GRAY);
	img.convertTo(img, CV_32FC1);
	img+= cv::Scalar::all(1);
	cv::log(img, img);
	cv::Scalar mean, stddev;
	cv::meanStdDev(img, mean,stddev);
	img = img - mean[0];
	img = img/(stddev[0] + 0.001);

	// multiply by hann window before dft 
	cv::Mat hann;
	cv::createHanningWindow(hann, img.size(), CV_32F);
	hann.convertTo(hann, CV_32FC1, 1/255.0);
	cv::multiply(img, hann, img);
}


// computes dft of given image and gives out resulting spectrum image
void ImageProcessor::computeDFT(cv::Mat& in, cv::Mat& out)
{
	cv::Mat gray;
	int x = cv::getOptimalDFTSize(in.rows);
	int y = cv::getOptimalDFTSize(in.cols);

	//cv::cvtColor(in, in, CV_RGB2GRAY);
	cv::copyMakeBorder(in,gray,0,x-in.rows,0,y-in.cols,cv::BORDER_CONSTANT,cv::Scalar::all(0));
	//in.copyTo(gray);
	cv::Mat planes[] = {cv::Mat_<float>(gray), cv::Mat::zeros(gray.size(), CV_32FC1)};
	cv::Mat complexImg;
	cv::merge(planes, 2, complexImg);

	dft(complexImg, complexImg);
	out = complexImg.clone(); // output 

	

	
}


// extracts patch of image for initialization
cv::Mat ImageProcessor::extractPatch(cv::Mat& in, Position& p)
{
	// extract rectangle from the image with given dimension with no rotation
	int x1 = 243;
	int y1 = 165;
	int w = 110;
	int h = 115;
	int x = x1-(w/2);
	int y = y1 - (h/2);
	//w = 2*w;
	//h = 2*h;
	cv::Mat window;
	extractRect(in, window, p.x,p.y,p.w,p.h); // left most corner and width and height. taken heuristically
	// resize the patch to a given dimension
	cv::Mat resizedImg;
	resizeImg(window,resizedImg);
	return resizedImg;
}

// reads all image filenames in given directory
void ImageProcessor::readDir()
{
	BOOST_FOREACH(boost::filesystem::path path,
            boost::make_iterator_range(
                boost::filesystem::recursive_directory_iterator(boost::filesystem::path("../vot15_car1/imgs")),
                boost::filesystem::recursive_directory_iterator()))
	{
        std::string s =  path.filename().string();
		std::stringstream ss(s);
    	std::string item;
    	std::vector<std::string> tokens;
    	while (getline(ss, item, '.')) {
         	tokens.push_back(item);

    	}
		std::string::size_type sz;
		long i_auto = std::stol (tokens[0],&sz);
		//std::cout<< tokens[0]<< ", "<<i_auto<<std::endl;
		_data_map[i_auto] = path.string();
    }
	//std::cout<<mymap.size();

}

void ImageProcessor::showDFT(cv::Mat& complexImg)
{
	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
	std::vector<cv::Mat> planes;
	cv::split(complexImg, planes);
	cv::magnitude(planes[0], planes[1], planes[0]);
	cv::Mat mag = planes[0];
	mag += cv::Scalar::all(1);
	cv::log(mag, mag);

	mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));
	int cx = mag.cols/2;
	int cy = mag.rows/2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	cv::Mat tmp;
	cv::Mat q0(mag, Rect(0, 0, cx, cy));
	cv::Mat q1(mag, Rect(cx, 0, cx, cy));
	cv::Mat q2(mag, Rect(0, cy, cx, cy));
	cv::Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);

	cv::imshow("spectrum magnitude", mag);
	cv::waitKey(5);
}

// shows response spectrum image
void ImageProcessor::showResponseImage(cv::Mat& img)
{
	cv::Mat res;
	cv::idft(img,res, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	res.convertTo(res, CV_32FC1, 1/255.0 );
	showImage(res);
}

void ImageProcessor::spectrumDiv(cv::Mat& a, cv::Mat& b, cv::Mat& out)
{
	CV_Assert(a.size() == b.size());
	out = cv::Mat::zeros(a.rows,a.cols,a.type());
	// iterate over whole matrix and compute each element
	for(int i = 0 ; i < a.rows; i++)
	{
		for(int j = 0; j < a.cols;j++)
		{
			out.at<cv::Vec2f>(i,j)[0] = ((a.at<cv::Vec2f>(i,j)[0]*b.at<cv::Vec2f>(i,j)[0]) + 
										(a.at<cv::Vec2f>(i,j)[1]*b.at<cv::Vec2f>(i,j)[1])) / 
										((b.at<cv::Vec2f>(i,j)[0]*b.at<cv::Vec2f>(i,j)[0]) + 
										(b.at<cv::Vec2f>(i,j)[1]*b.at<cv::Vec2f>(i,j)[1])); 

			out.at<cv::Vec2f>(i,j)[1] = ((a.at<cv::Vec2f>(i,j)[0]*b.at<cv::Vec2f>(i,j)[0]) - 
										(a.at<cv::Vec2f>(i,j)[1]*b.at<cv::Vec2f>(i,j)[1])) / 
										((b.at<cv::Vec2f>(i,j)[0]*b.at<cv::Vec2f>(i,j)[0]) + 
										(b.at<cv::Vec2f>(i,j)[1]*b.at<cv::Vec2f>(i,j)[1])); 
		}	
	}	

}


// runs algorithm
void ImageProcessor::run()
{
	
	readDir(); // read all data and store it to dictionary



	// initialize parameters used in multiple iterations
	cv::Mat prev_img, curr_img;
	//ModelH h_hat;
	ModelH curr_h_hat;
	cv::Mat phi_hat;
	cv::Mat h_hat_num;
	cv::Mat h_hat_den;
	int cx, cy; // postion of the object center
	
	// create desired output response y
	cv::Mat y = cv::Mat::zeros(_fixed_patch_size, _fixed_patch_size, CV_32FC1);
	y.at<float>(_fixed_patch_size/2, _fixed_patch_size/2) = 1.0f;
	cv::GaussianBlur(y,y, cv::Size(-1,-1),_fixed_patch_size/16,0);
	cv::normalize(y,y,cv::NORM_MINMAX);

	// compute dft for desired output response y
	cv::Mat y_hat;
	cv::dft(y,y_hat, cv::DFT_COMPLEX_OUTPUT );
	//showDFT(y_hat);
	// main loop 
	for( auto it : _data_map)
	{
		//std::cout<<it.first<<" "<<it.second<<std::endl;
		if(it.first == 1)
		{
			// for the first frame initialize tracker 
			curr_img = cv::imread(it.second, CV_LOAD_IMAGE_COLOR);
			CV_Assert(curr_img.channels() == 1 || curr_img.channels() == 3);
			cv::Mat resizedImg  = extractPatch(curr_img, _p);
			
			std::cout << "prepocessing input image "<< std::endl;
			preprocessImg(resizedImg);

			//computeH(resizedImg, curr_h_hat);
			std::cout<< "computing DFT" << std::endl;
			computeDFT(resizedImg, phi_hat);
			//showDFT(phi_hat);
			std::cout<<"phi hat : "<<phi_hat.size()<< std::endl;

			convolveDFT(y_hat, phi_hat, h_hat_num);
			convolveDFT(phi_hat, phi_hat, h_hat_den);
			// initialize position
			cx = _fixed_patch_size/2;
			cy = _fixed_patch_size/2;

			//curr_img.copyTo(prev_img);

		}
		else
		{
			curr_img = cv::imread(it.second, CV_LOAD_IMAGE_COLOR);
			Position new_p;
			new_p.x = _p.x-(_p.w/2);
			new_p.y = _p.y - (_p.h/2);
			new_p.w = 2*_p.w;
			new_p.h = 2*_p.h;
			cv::Mat resizedImg  = extractPatch(curr_img, new_p);
			std::cout<<"input image size: "<< resizedImg.size()<< std::endl;

			std::cout << "prepocessing input image "<< std::endl;
			preprocessImg(resizedImg);

			computeDFT(resizedImg, phi_hat);
			//showDFT(phi_hat);
			std::cout<<"phi hat size: "<< phi_hat.size()<< std::endl;
			// add lambda to denominator 
			cv::Mat lambda = cv::Mat::eye(h_hat_den.size(), h_hat_den.type());
			lambda = _reg_param*lambda;
			h_hat_den += lambda;
			// compute over filter 
			cv::Mat h_hat;
			spectrumDiv(h_hat_num, h_hat_den, h_hat);

			// convolve regularized filter with current image and compute ifft 
			cv::Mat phi_hat_resp;
			convolveDFT(phi_hat, h_hat, phi_hat_resp);

			// compute max loc
			 cv::Point max_loc;
			// cv::minMaxLoc(phi_hat, NULL, NULL, NULL, &maxLoc);

			// compute inverse fourier transform
			cv::Mat phi;
			cv::dft(phi_hat_resp, phi, DFT_INVERSE | DFT_REAL_OUTPUT);
			cv::normalize(phi, phi, 0.0, 255.0, NORM_MINMAX);
			showImage(phi);
			// resize 
			//cv::resize(phi,phi, cv::Size(_p.w, _p.h));
			//cv::Point maxLoc;
			cv::minMaxLoc(phi, NULL, NULL, NULL, &max_loc);
			int xd = max_loc.x - cx ;
			int yd = max_loc.y - cy;

			std::cout<< max_loc.x<< ", "<< max_loc.y<< std::endl;

			// filter update 
			cv::Mat new_h_hat_num;
			cv::Mat new_h_hat_den;
			convolveDFT(y_hat, phi_hat, new_h_hat_num);
			convolveDFT(phi_hat, phi_hat, new_h_hat_den);

			h_hat_num = (1-_templ_learning_rate)*h_hat_num + _templ_learning_rate*new_h_hat_num;
			h_hat_den = (1-_templ_learning_rate)*h_hat_den + _templ_learning_rate*new_h_hat_den;

			// update rect position
			_p.x +=xd;
			_p.y +=yd;
			cx += xd;
			cy += yd; 
			// compute max location

			//showDFT(h_hat);

			/*
			// apply hann window before 
			cv::Mat hann;
			cv::createHanningWindow(hann, cv::Size(_fixed_patch_size, _fixed_patch_size), CV_32F);
			hann.convertTo(hann, CV_32FC1, 1/255.0);

			cv::Mat phi; // feature image (currently grayscale image)
			cv::cvtColor(resizedImg, phi, CV_RGB2GRAY);
			cv::equalizeHist(phi, phi); // histogram equaizer for more contrast in image features
			phi.convertTo(phi, CV_32FC1,1/255.0 );

			//apply hann window before fourier transform
			//cv::multiply(phi, hann, phi);
			phi = phi*hann;
			//showImage(phi);
			// take fourier transform of feature image
			cv::Mat phi_hat;
			cv::dft(phi,phi_hat, cv::DFT_COMPLEX_OUTPUT );

			// compute filter 
			//cv::Mat h = (1/(cv::trace(curr_h_hat.s)[0] + 1e-3))*curr_h_hat.r;
			std::cout<< cv::trace(curr_h_hat.s)<<std::endl;
			std::vector<cv::Mat> components;
			cv::split(curr_h_hat.r, components);
			//cv::Mat h_real = components[0];
			components[0] = (1/(cv::trace(curr_h_hat.s)[0] +1e-3))*components[0];
			//cv::Mat h_imag = components[1];
			components[1] = (1/(cv::trace(curr_h_hat.s)[1] + 1e-3))*components[1];
			cv::Mat h;
			cv::merge(components,h);

			// compute response in frequency domain	
			cv::Mat response_hat;
			cv::mulSpectrums(h, phi_hat, response_hat,0,true);	
			std::cout<< h.size()<< ", "<< h.type()<<std::endl;
			//showResponseImage(response_hat);
			//showResponseImage(response_hat);
			cv::Mat res;
			cv::idft(response_hat,res, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
			std::cout<< res.size()<< ", "<< res.type()<<std::endl;
			cv::Point minLoc, maxLoc;
			double minVal,maxVal;
			cv::minMaxLoc(res, &minVal,&maxVal, &minLoc, &maxLoc);
			std::cout<< minVal << ", "<< maxVal<< ", " << std::endl;
			//showImage(res);
			//computeH(resizedImg, curr_h_hat);
			*/
		}
	}


}

int main(int argc, char **argv)
{
	ImageProcessor processor;

	// processor.initializeImages("../vot15_car1/imgs/00000001.jpg");
	// processor.setCurrentImage("../vot15_car1/imgs/00000002.jpg");
	processor.run();

	return 0;
}
