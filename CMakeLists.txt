set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED "ON")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Add useful compiler flags.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfloat-conversion -Wmissing-braces")

cmake_minimum_required(VERSION 3.12.0)
project(NSDS_MPI LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MPI package
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(average src/average.c)
target_link_libraries(average MPI::MPI_C)
add_executable(char_count src/char_count.c)
target_link_libraries(char_count MPI::MPI_C)
add_executable(filter src/filter.c)
target_link_libraries(filter MPI::MPI_C)
add_executable(ping_pong src/ping_pong.c)
target_link_libraries(ping_pong MPI::MPI_C)
add_executable(probe src/probe.c)
target_link_libraries(probe MPI::MPI_C)
add_executable(recv src/recv.c)
target_link_libraries(recv MPI::MPI_C)
add_executable(ring src/ring.c)
target_link_libraries(ring MPI::MPI_C)
