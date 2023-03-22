project(NcnnYolov8)


add_executable(${PROJECT_NAME}_test
        NcnnYolov8Test.cpp)

target_include_directories(${PROJECT_NAME}_test
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include
        )

target_link_libraries(${PROJECT_NAME}_test
        PRIVATE
        OIYolo)