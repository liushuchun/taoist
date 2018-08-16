cmake_minimum_required(VERSION 3.8)
project(taoist)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v -fPIC -lstdc++ -funroll-loops -Wall -std=c++11 -Ofast -g -DEIGEN_FAST_MATH -march=native")

message("taoist building")
set(CMAKE_CXX_STANDARD 11)
include_directories("/usr/local/include/eigen3/")
aux_source_directory(./src DIR_SRCS)

add_subdirectory(tests)
set(SOURCE_FILES main.cpp)
add_executable(taoist ${SOURCE_FILES} ${DIR_SRCS})

