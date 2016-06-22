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


## Warnings 
* Compiler should support c++11 flags. 
* Tested only on Unix/Linux systems

## TODO

* Add Scale tracker to estimate scale of the object robustly. 
* Use HOG features for making it more robust. 

## Contact

* Abhinav Dadhich