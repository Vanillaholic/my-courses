cmake_minimum_required(VERSION 3.29)
project(add_reverb)

set(CMAKE_CXX_STANDARD 17)

add_executable(add_reverb main.cpp baselib.cpp conv.cpp)
