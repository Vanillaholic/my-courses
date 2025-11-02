cmake_minimum_required(VERSION 3.29)
project(HW4_code)

set(CMAKE_CXX_STANDARD 20)

# 设置可执行文件输出到项目根目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(LMS main.cpp adapt_filt.cpp)
