project(NcnnYolov8)

if(NOT EMBARCADERO)
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

if (EMBARCADERO)
	message(STATUS "NCNN_ROOT=${NCNN_ROOT}")
	target_link_libraries(${PROJECT_NAME}_test1
			PRIVATE
			OIYolo
			${NCNN_ROOT}\\lib\\ncnn${CMAKE_STATIC_LIBRARY_SUFFIX}) # TODO: Hack for Embarcadero
else()
	target_link_libraries(${PROJECT_NAME}_test1
			PRIVATE
			OIYolo
			-L${NCNN_ROOT}/lib libncnn.a) # TODO: Hack for MinGW
endif()

