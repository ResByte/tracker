#include <iostream>
#include "tracker/image_processor.hpp"

int main(int argc, char **argv)
{
	Parameters param;
	ImageProcessor processor(param);

	//starts the process
	processor.run();

	return 0;
}
