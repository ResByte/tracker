/*
 * histogram_model.cpp
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


class HistogramModel{
public:
	
	HistogramModel();
	
	void updateHistModel();
	
	// creates a greyscale histogram of an image patch
	void computeHistogram();
	
	// new model
	
	// patch
	
	// background area
	
	// foreground area 
	
	// target size 
	
	// norm area 
	
	// number of bins
	
	// greysacle sequence 
	
	// background histogram 
	
	// foreground histogram 
	
	// learning rate pwp 
	
		
	
};

void HistogramModel::updateHistModel()
{
	
}


void HistogramModel::computeHistogram()
{
	
}	
