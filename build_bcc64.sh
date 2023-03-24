#!/bin/bash

export PATH=$PATH:"/c/ninja-win"

build_type=Debug
if [ -z "$1" ]; then
  echo "build_type='$build_type'";
else
  build_type=$1
  echo "build_type='$build_type'";
fi

git clone https://github.com/Tencent/ncnn.git
cd ncnn

cmake -Bbuild_bcc64\
 -G "Ninja"\
 -DCMAKE_C_COMPILER=bcc64.exe\
 -DCMAKE_CXX_COMPILER=bcc64.exe\
 -DCMAKE_BUILD_TYPE=$build_type\
 -DCMAKE_VERBOSE_MAKEFILE=ON\
 -DCMAKE_C_COMPILER_WORKS=ON\
 -DCMAKE_CXX_COMPILER_WORKS=ON\
 -DNCNN_BUILD_TOOLS=OFF\
 -DNCNN_BUILD_EXAMPLES=OFF\
 -DNCNN_BUILD_BENCHMARK=OFF\
 -DNCNN_AVX=OFF\
 -DNCNN_AVX2=OFF\
 -DNCNN_SSE2=OFF\
 -DNCNN_RUNTIME_CPU=OFF\
 -DNCNN_THREADS=OFF\
 -DNCNN_OPENMP=OFF\
 -DNCNN_XOP=OFF\
 -DNCNN_PIXEL_AFFINE=OFF\
 -DNCNN_INSTALL_SDK=ON\
 -DCMAKE_INSTALL_PREFIX=$(pwd)/../OIYoloDependencies64\
 -DDEPENDENCIES_PREFIX_PATH=$(pwd)/../OIYoloDependencies64
cp ../patches/cpuid.h ./src/

cmake --build build_bcc64 --target install
cd ..

#--trace-expand\
cmake -Bbuild_bcc64\
 -G "Ninja"\
 -DCMAKE_BASE_NAME=bcc64\
 -DCMAKE_C_COMPILER=bcc64.exe\
 -DCMAKE_CXX_COMPILER=bcc64.exe\
 -DCMAKE_BUILD_TYPE=$build_type\
 -DCMAKE_VERBOSE_MAKEFILE=ON\
 -DCMAKE_C_COMPILER_WORKS=ON\
 -DCMAKE_CXX_COMPILER_WORKS=ON\
 -DEMBARCADERO=ON\
 -DNCNN_ROOT=$(pwd)/OIYoloDependencies64\
 -DCMAKE_INSTALL_PREFIX=$(pwd)/OIYolo_Install64
cmake --build build_bcc64 --target install

./build_bcc64/test/NcnnYolov8_test1.exe ./assets/parking.jpg ./assets/yolov8s.param ./assets/yolov8s.bin ./assets/yolov8s.classes 640
