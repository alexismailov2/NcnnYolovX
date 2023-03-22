#!/bin/bash

build_type=Debug
if [ -z "$1" ]; then
  echo "build_type='$build_type'";
else
  build_type=$1
  echo "build_type='$build_type'";
fi

cmake -Bbuild_bcc32\
 -G "Borland Makefiles"\
 -DCMAKE_MAKE_PROGRAM="/c/Program Files (x86)/Embarcadero/Studio/21.0/bin/make.exe"\
 -DCMAKE_C_COMPILER=bcc32x.exe\
 -DCMAKE_CXX_COMPILER=bcc32x.exe\
 -DCMAKE_BUILD_TYPE=$build_type\
 -DCMAKE_VERBOSE_MAKEFILE=ON\
 -DCMAKE_C_COMPILER_WORKS=ON\
 -DCMAKE_CXX_COMPILER_WORKS=ON
#should be moved to ExternalAdd project
#cp ../cpuid.h ./src/
#sed -i 's:/:\\:g' ./build_bcc32/src/CMakeFiles/ncnn.dir/objects1.rsp
cmake --build build_bcc32 --target install