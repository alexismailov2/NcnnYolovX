if("${DEPENDENCIES_PREFIX_PATH}" STREQUAL "")
    set(DEPENDENCIES_PREFIX_PATH ${CMAKE_SOURCE_DIR}/OIYoloDependencies)
endif()
message(STATUS DEPENDENCIES_PREFIX_PATH=${DEPENDENCIES_PREFIX_PATH})

if(EXISTS ${DEPENDENCIES_PREFIX_PATH}/libncnn)
    set(NCNN_ROOT ${DEPENDENCIES_PREFIX_PATH}/libncnn)
    message(STATUS NCNN_ROOT=${NCNN_ROOT})
endif()

include(ExternalProject)
include(cmake/ExternalNcnn.cmake)