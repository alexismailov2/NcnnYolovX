#!/bin/bash

build_type=Debug
if [ -z "$1" ]; then
  echo "build_type='$build_type'";
else
  build_type=$1
  echo "build_type='$build_type'";
fi

git clone https://github.com/Tencent/ncnn.git
cd ncnn

cmake -Bbuild_embarc\
 -G "Borland Makefiles"\
 -DCMAKE_MAKE_PROGRAM="/c/Program Files (x86)/Embarcadero/Studio/21.0/bin/make.exe"\
 -DCMAKE_C_COMPILER=bcc32c.exe\
 -DCMAKE_CXX_COMPILER=bcc32c.exe\
 -DCMAKE_BUILD_TYPE_INIT=Debug\
 -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON\
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
 -DNCNN_INSTALL_SDK=ON
cp ../patches/cpuid.h ./src/
sed -i 's:/:\\:g' ./build_embarc/src/CMakeFiles/ncnn.dir/objects1.rsp
sed -i 's:-Wall: :g' "./build_embarc/src/CMakeFiles/ncnn.dir/flags.make"
sed -i 's:-Wextra: :g' "./build_embarc/src/CMakeFiles/ncnn.dir/flags.make"
sed -i 's:-Wno-unused-function: :g' "./build_embarc/src/CMakeFiles/ncnn.dir/flags.make"
sed -i 's:-Ofast: :g' "./build_embarc/src/CMakeFiles/ncnn.dir/flags.make"
sed -i 's:-fvisibility=hidden: :g' "./build_embarc/src/CMakeFiles/ncnn.dir/flags.make"
sed -i 's:-fvisibility-inlines-hidden: :g' "./build_embarc/src/CMakeFiles/ncnn.dir/flags.make"

cmake --build build_embarc --target install
cd ..

cmake -Bbuild_bcc32\
 -G "Borland Makefiles"\
 -DCMAKE_MAKE_PROGRAM="/c/Program Files (x86)/Embarcadero/Studio/21.0/bin/make.exe"\
 -DCMAKE_C_COMPILER=bcc32c.exe\
 -DCMAKE_CXX_COMPILER=bcc32c.exe\
 -DCMAKE_BUILD_TYPE=$build_type\
 -DCMAKE_VERBOSE_MAKEFILE=ON\
 -DCMAKE_C_COMPILER_WORKS=ON\
 -DCMAKE_CXX_COMPILER_WORKS=ON\
 -DEMBARCADERO=ON\
 -DNCNN_ROOT=$(pwd)/ncnn/build_embarc/install\
 -DCMAKE_INSTALL_PREFIX=./Install
sed -i 's:-Od: :g' "./build_bcc32/CMakeFiles/OIYolo.dir/flags.make"
sed -i 's:-Od: :g' "./build_bcc32/CMakeFiles/OIYolo.dir/build.make"
sed -i 's:-Od: :g' "./build_bcc32/test/CMakeFiles/NcnnYolov8_test1.dir/flags.make"
sed -i 's:-Od: :g' "./build_bcc32/test/CMakeFiles/NcnnYolov8_test1.dir/build.make"
cmake --build build_bcc32 --target install

./build_bcc32/test/NcnnYolov8_test1.exe ./assets/parking.jpg ./assets/yolov8s.param ./assets/yolov8s.bin ./assets/yolov8s.classes 640
