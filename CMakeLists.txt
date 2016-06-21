cmake_minimum_required(VERSION 2.8)
PROJECT( opencv_test )
set(CMAKE_CXX_FLAGS "-std=c++11")
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system filesystem )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} include)



add_library(proj_libs
	src/image_processor.cpp
)

target_link_libraries(proj_libs ${OpenCV_LIBS} ${Boost_LIBRARIES})


ADD_EXECUTABLE( tracker src/tracker.cpp )
TARGET_LINK_LIBRARIES(tracker proj_libs )
