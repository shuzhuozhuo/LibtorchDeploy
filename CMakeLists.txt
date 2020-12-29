cmake_minimum_required(VERSION 3.10.2)
project(PytorchDeploy)

set(Torch_DIR /home/shu/APP/libtorch/share/cmake/Torch)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

if(NOT Torch_FOUND)
	message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)
    
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
message("OPENCV DIR: " ${OpenCV_INCLUDE_DIRS})
message(libraries: 1 ${TORCH_LIBARRIES})



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(include)
aux_source_directory(src MySrc)

# name of executable file and path of source file
add_executable(PytorchDeploy main.cpp ${MySrc})
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(PytorchDeploy ${OpenCV_LIBS} "${TORCH_LIBRARIES}")  
set_property(TARGET PytorchDeploy PROPERTY CXX_STANDARD 14)