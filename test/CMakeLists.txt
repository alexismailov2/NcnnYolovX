project(NcnnYolov8)

if(OIYolo_OpenCV)
add_executable(${PROJECT_NAME}_test
        NcnnYolov8Test.cpp)

target_include_directories(${PROJECT_NAME}_test
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include)

target_link_libraries(${PROJECT_NAME}_test
        PRIVATE
        OIYolo)
endif()

add_executable(${PROJECT_NAME}_test1
        NcnnYolov8Test1.cpp)

target_include_directories(${PROJECT_NAME}_test1
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include)

target_link_libraries(${PROJECT_NAME}_test1
        PRIVATE
        OIYolo
        ${NCNN_ROOT}/lib/libncnn.a) # TODO: Hack for MinGW