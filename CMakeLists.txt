# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.0)
project(L_slam LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11) # C++11...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS ON) #...with compiler extensions like gnu++11

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

find_package( OpenCV REQUIRED )
#set(aruco_DIR "/home/local /share/aruco/cmake")
#find_package( aruco REQUIRED )
set(aruco_INCLUDE_DIRS  /usr/local/include/aruco)
set(aruco_LIBS  /usr/local/lib/libaruco.so.3.1.15)

include_directories( include ${OpenCV_INCLUDE_DIRS} ${aruco_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${OpenCV_LIBS}  ${aruco_LIBS} )

include_directories(3rdparty/eigen3)

IF(NOT USE_OWN_EIGEN3)
    find_package( Eigen3 REQUIRED )
ELSE()
    SET(EIGEN3_INCLUDE_DIR "3rdparty/eigen3")
ENDIF()

add_executable( cam_localization test/cam_localization.cpp )
target_link_libraries( cam_localization ${REQUIRED_LIBRARIES} )

# If could not find could not find a package configuration file provided by "aruco",
# set the "aruco_INCLUDE_DIRS" and the "aruco_LIBS" by mabual setting directly.
#set(aruco_INCLUDE_DIRS  /home/exbot/OpenCV/aruco306-installed/include/aruco)
#set(aruco_LIBS  /home/exbot/OpenCV/aruco306-installed/lib/libaruco.so)

#include_directories( include ${OpenCV_INCLUDE_DIRS} ${aruco_INCLUDE_DIRS})
