# Video Object Tracking
This is a C++ implementation(in progress) of STAPLE:Complementary Learners for Real Time Tracking Object[1] by Bertinetto et al. 

	@article{bertinetto2015staple,
	  title={Staple: Complementary Learners for Real-Time Tracking},
	  author={Bertinetto, Luca and Valmadre, Jack and Golodetz, Stuart and Miksik, Ondrej and Torr, Philip},
	  journal={arXiv preprint arXiv:1512.01355},
	  year={2015}
	}

STAPLE provides a combination of learners to track robustly an object in a video with color changes as well as deformations. It uses a combination of independent ridge regression solver for real time performance. Correlation filter and color histogram based tracking scores are combined in online fashion to get robust position and scale of object at about 80FPS. The results are tested on VOT benchmark dataset. 


In the current version, Template filter based on discriminative correlation filter is implemented. This combines online tracking of the target object center as well as learning the parameters for tracking.  This is one part of the STAPLE. Currently, we are using grayscale values as features for the input image. 


## Dependencies

* cmake
* Opencv-2.4
* boost


## Installation and Execution

* clone/download this repository
* inside tracker folder, `mkdir build && cd build`
* `cmake ..`
* `make`
* `./tracker`
* frame with car (surrounded by ractangular box is displayed)
* click on the image frame and press any key to update to next data frames.

## Warnings 

* Compiler should support c++11. 
* Tested only on Linux systems

## Results

* Results are displayed as rectangular box surrounding target object. 

## Performance

Tracker parameters are fixed as in the paper. Using only translation based tracker, performance is not upto mark. Tracker tracks robustly for initial few frames but after recieving distorted images, tracker loses the object and only in last few frames tracker is able to track given object again.


## TODO

* Create evaluation tools for correlation filter.
* Add Scale tracker to estimate scale of the object robustly. 
* Use HOG features for making it more robust. 

## Contact

* Abhinav Dadhich