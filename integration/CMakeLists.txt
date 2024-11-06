cmake_minimum_required(VERSION 3.12)
project(my_project)

find_package(pybind11 REQUIRED)

add_executable(my_executable main.cpp)
target_link_libraries(my_executable PRIVATE pybind11::embed)  # For embedding Python
