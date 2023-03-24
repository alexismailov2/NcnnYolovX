include(ExternalProject)

if("${DEPENDENCIES_PREFIX_PATH}" STREQUAL "")
    set(DEPENDENCIES_PREFIX_PATH ${CMAKE_SOURCE_DIR}/OIYoloDependencies)
endif()
message(STATUS DEPENDENCIES_PREFIX_PATH=${DEPENDENCIES_PREFIX_PATH})

if("${NCNN_ROOT}" STREQUAL "")
    set(NCNN_ROOT ${DEPENDENCIES_PREFIX_PATH}/libncnn)
    message(STATUS NCNN_ROOT=${NCNN_ROOT})
    if(NOT EXISTS ${NCNN_ROOT})
        include(cmake/ExternalNcnn.cmake)
    endif()
endif()
