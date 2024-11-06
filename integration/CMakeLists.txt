cmake_minimum_required(VERSION 3.12)
project(attn_module)

add_subdirectory(pybind11)
pybind11_add_module(attn_module cpp/attn_module.cpp)
set_target_properties(attn_module PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(attn_module PRIVATE pybind11::module)

target_compile_options(attn_module PRIVATE -mavx2)  # Enable AVX2 for SIMD
target_compile_options(attn_module PRIVATE -march=native)  # Enable AVX2 for SIMD
target_compile_options(attn_module PRIVATE -lnuma) 
target_compile_options(attn_module PRIVATE -ffast-math) 
target_compile_options(attn_module PRIVATE -mfma) 
target_compile_options(attn_module PRIVATE -O3) 
target_compile_options(attn_module PRIVATE -pedantic)