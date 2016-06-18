cmake_minimum_required(VERSION 2.8)
PROJECT( opencv_test )
set(CMAKE_CXX_FLAGS "-std=c++11")
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system filesystem )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

ADD_EXECUTABLE( image_processor src/image_processor.cpp )
TARGET_LINK_LIBRARIES(image_processor ${OpenCV_LIBS} ${Boost_LIBRARIES})
