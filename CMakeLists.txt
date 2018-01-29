cmake_minimum_required(VERSION 3.8)
project(taoist)
message("taoist building")
set(CMAKE_CXX_STANDARD 11)
include_directories("/usr/local/include/eigen3/")
aux_source_directory(./src DIR_SRCS)
set(SOURCE_FILES main.cpp)
add_executable(taoist ${SOURCE_FILES} ${DIR_SRCS})


