message(STATUS "Ncnn will be build for ${CMAKE_SYSTEM_NAME}")

ExternalProject_Add(external_ncnn
        GIT_REPOSITORY    https://github.com/Tencent/ncnn.git
        GIT_SHALLOW       ON
        GIT_PROGRESS      ON
        CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${DEPENDENCIES_PREFIX_PATH}/libncnn
                          -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
                          -DCMAKE_BUILD_TYPE_INIT=${CMAKE_BUILD_TYPE}
                          -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
                          -DCMAKE_C_COMPILER_WORKS=ON
                          -DCMAKE_CXX_COMPILER_WORKS=ON
                          -DNCNN_BUILD_TOOLS=OFF
                          -DNCNN_BUILD_EXAMPLES=OFF
                          -DNCNN_BUILD_BENCHMARK=OFF
                          -DNCNN_AVX=OFF
                          -DNCNN_AVX2=OFF
                          -DNCNN_SSE2=OFF
                          -DNCNN_RUNTIME_CPU=OFF
                          -DNCNN_THREADS=OFF
                          -DNCNN_OPENMP=OFF
                          -DNCNN_XOP=OFF
                          -DNCNN_PIXEL_AFFINE=OFF
                          -DNCNN_INSTALL_SDK=ON)

set(NCNN_ROOT ${DEPENDENCIES_PREFIX_PATH}/libncnn)
list(APPEND DEPENDENCIES_LIST external_ncnn)

message(STATUS DEPENDENCIES_LIST="${DEPENDENCIES_LIST}")
add_custom_target(dependencies DEPENDS ${DEPENDENCIES_LIST})
