#!/bin/bash

build_type=Debug
if [ -z "$1" ]; then
  echo "build_type='$build_type'";
else
  build_type=$1
  echo "build_type='$build_type'";
fi

cmake -Bbuild_host\
 -DDEPENDENCIES_PREFIX_PATH=$(pwd)/build_host\
 -DCMAKE_BUILD_TYPE=$build_type\
 -DCMAKE_VERBOSE_MAKEFILE=ON\
 -DCMAKE_C_COMPILER_WORKS=ON\
 -DCMAKE_CXX_COMPILER_WORKS=ON
# -DCMAKE_INSTALL_PREFIX=$(pwd)/OIYoloInstall
cmake --build build_host --target install