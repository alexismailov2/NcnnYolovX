cmake_minimum_required(VERSION 3.25)
project(OIYolo)

if (NOT CMAKE_INSTALL_PREFIX)
	set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(CMAKE_CXX_STANDARD 14)

option(OIYolo_INSTALL "install ncnn library and headers" ON)
option(OIYolo_OpenCV "use OpenCV library for inference" OFF)
option(OIYolo_NCNN "use NCNN library for inference" ON)
option(EMBARCADERO "For Embarcadero there is workarounds" OFF)

if (OIYolo_OpenCV AND OIYolo_NCNN)
	message(FATAL_ERROR "Only one engine supported at the same time")
endif()

include(cmake/Dependencies.cmake)

add_library(${PROJECT_NAME}
		src/ncnn_port/V8.cpp)

if (DEFINED ${DEPENDENCIES_LIST})
	add_dependencies(${PROJECT_NAME} ${DEPENDENCIES_LIST})
endif()

target_include_directories(${PROJECT_NAME}
		PUBLIC
		${CMAKE_CURRENT_LIST_DIR}/include)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
	if (OIYolo_OpenCV)
		find_package(OpenCV REQUIRED)
		add_definitions(-DOIYolo_OpenCV=${OIYolo_OpenCV})
		target_include_directories(${PROJECT_NAME}
				PUBLIC
				${OpenCV_INCLUDE_DIRS})
		target_link_libraries(${PROJECT_NAME}
				PUBLIC
				${OpenCV_LIBS})
	endif()

	if (OIYolo_NCNN)
		message(STATUS NCNN_ROOT=${NCNN_ROOT})
		target_include_directories(${PROJECT_NAME}
				PUBLIC
				${NCNN_ROOT}/include)

		target_link_libraries(${PROJECT_NAME}
				PUBLIC
				${NCNN_ROOT}/lib/libncnn.a)
	endif()
else()
	if (OIYolo_NCNN)
		message(STATUS "CMAKE_STATIC_LIBRARY_SUFFIX=${CMAKE_STATIC_LIBRARY_SUFFIX}")
		target_include_directories(${PROJECT_NAME}
				PRIVATE
				${NCNN_ROOT}\\include)

		target_link_libraries(${PROJECT_NAME}
				PRIVATE
				-L${NCNN_ROOT}\\lib ncnn${CMAKE_STATIC_LIBRARY_SUFFIX})
	else()
		message(FATAL_ERROR "OpenCV supported currently only on Darwin/Linux systems")
	endif()
endif()

include(GNUInstallDirs)
message(STATUS CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})
message(STATUS CMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR})
message(STATUS CMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ${PROJECT_NAME} EXPORT OIYolo
		ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
		RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
		)
file(GLOB OIYolo_PUBLIC_INCLUDES ${CMAKE_CURRENT_LIST_DIR}/include/OIYolo/*.hpp)
install(FILES
		${OIYolo_PUBLIC_INCLUDES}
		DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/OIYolo/)

add_subdirectory(test)
