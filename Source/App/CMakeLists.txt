#
# Copyright(c) 2019 Intel Corporation
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
#

cmake_minimum_required(VERSION 3.5...3.28)

# APP Directory CMakeLists.txt

# Include Subdirectories
include_directories(${PROJECT_SOURCE_DIR}/Source/API/)
find_package(PkgConfig)
if(NOT PkgConfig_FOUND)
  message(WARNING "PkgConfig not found; disabled building with dovi & hdr10plus.")
endif()

# libdovi detection & preprocessor macro
option(LIBDOVI_FOUND "Use the dovi library" OFF)
if(LIBDOVI_FOUND)
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    find_path(PKG_CONFIG_FOUND NAMES dovi HINTS "${PROJECT_SOURCE_DIR}/Source/dovi/msvc/lib")
    find_library(LIBDOVI_LIBRARY NAMES dovi.lib HINTS "${PROJECT_SOURCE_DIR}/Source/dovi/msvc/lib")
    find_path(LIBDOVI_INCLUDE_DIR NAMES libdovi/rpu_parser.h HINTS "${PROJECT_SOURCE_DIR}/Source/dovi/msvc/include")
  elseif (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    find_path(PKG_CONFIG_FOUND NAMES dovi HINTS "${PROJECT_SOURCE_DIR}/Source/dovi/gcc/lib")
    find_library(LIBDOVI_LIBRARY NAMES dovi libdovi HINTS "${PROJECT_SOURCE_DIR}/Source/dovi/gcc/lib")
    find_path(LIBDOVI_INCLUDE_DIR NAMES libdovi/rpu_parser.h HINTS "${PROJECT_SOURCE_DIR}/Source/dovi/gcc/include")
  endif()

  if(NOT LIBDOVI_LIBRARY)
  message(FATAL_ERROR "dovi library not found!")
  endif()

  set(LIBDOVI_INCLUDE_DIRS ${LIBDOVI_INCLUDE_DIR})
  set(LIBDOVI_LIBRARIES ${LIBDOVI_LIBRARY})
  mark_as_advanced(LIBDOVI_INCLUDE_DIR LIBDOVI_LIBRARY)
  add_definitions(-DLIBDOVI_FOUND=1)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with dovi support - Yes\n")
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with dovi support - No\n")
endif()

# libhdr10plus detection & preprocessor macro
option(LIBHDR10PLUS_RS_FOUND "Use the hdr10plus library" OFF)
if(LIBHDR10PLUS_RS_FOUND)
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    find_path(PKG_CONFIG_FOUND NAMES hdr10plus-rs HINTS "${PROJECT_SOURCE_DIR}/Source/hdr/msvc/lib")
    find_library(LIBHDR10PLUS_RS_LIBRARY NAMES hdr10plus-rs.lib libhdr10plus-rs.lib HINTS "${PROJECT_SOURCE_DIR}/Source/hdr/msvc/lib")
    find_path(LIBHDR10PLUS_RS_INCLUDE_DIR NAMES libhdr10plus-rs/hdr10plus.h HINTS "${PROJECT_SOURCE_DIR}/Source/hdr/msvc/include")
  elseif (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    find_path(PKG_CONFIG_FOUND NAMES hdr10plus-rs HINTS "${PROJECT_SOURCE_DIR}/Source/hdr/gcc/lib")
    find_library(LIBHDR10PLUS_RS_LIBRARY NAMES hdr10plus-rs libhdr10plus-rs HINTS "${PROJECT_SOURCE_DIR}/Source/hdr/gcc/lib")
    find_path(LIBHDR10PLUS_RS_INCLUDE_DIR NAMES libhdr10plus-rs/hdr10plus.h HINTS "${PROJECT_SOURCE_DIR}/Source/hdr/gcc/include")
  endif()

  if(NOT LIBHDR10PLUS_RS_LIBRARY)
  message(FATAL_ERROR "hdr10plus library not found!")
  endif()

  set(LIBHDR10PLUS_RS_INCLUDE_DIRS ${LIBHDR10PLUS_RS_INCLUDE_DIR})
  set(LIBHDR10PLUS_RS_LIBRARIES ${LIBHDR10PLUS_RS_LIBRARY})
  mark_as_advanced(LIBHDR10PLUS_RS_INCLUDE_DIR LIBHDR10PLUS_RS_LIBRARY)
  add_definitions(-DLIBHDR10PLUS_RS_FOUND=1)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with hdr10plus support - Yes\n")
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with hdr10plus support - No\n")
endif()

# mimalloc detection & procession macro
option(MIMALLOC_FOUND "Use mimalloc library" OFF)
if(MIMALLOC_FOUND)
  find_library(MIMALLOC_LIB mimalloc.dll.lib HINTS "${PROJECT_SOURCE_DIR}/../mimalloc/mi_build/Release")
  find_path(MIMALLOC_INCLUDE_DIR mimalloc.h HINTS "${PROJECT_SOURCE_DIR}/../mimalloc/include")
  find_path(MIMALLOC_DLL mimalloc.dll HINTS "${PROJECT_SOURCE_DIR}/../mimalloc/mi_build/Release")
  find_path(MIMALLOC_REDIRECT_DLL mimalloc-redirect.dll HINTS "${PROJECT_SOURCE_DIR}/../mimalloc/mi_build/Release")
  include_directories(${MIMALLOC_INCLUDE_DIR})
   #find_library(MIMALLOC_LIB mimalloc.lib HINTS "${PROJECT_SOURCE_DIR}/../mimalloc/mi_build/Release")
# Set static linking for mimalloc
#  set(MIMALLOC_USE_STATIC_LIBS ON)
  if(MIMALLOC_LIB)
    message(STATUS "Found mimalloc: ${MIMALLOC_LIB}")
  else()
    message(FATAL_ERROR "mimalloc library not found!")
  endif()
  add_definitions(-DMIMALLOC_FOUND=1)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with mimalloc support - Yes\n")
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with mimalloc support - No\n")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BUILD_TYPE_DIR "Debug")
else()
    set(BUILD_TYPE_DIR "Release")
endif()

if (MIMALLOC_FOUND AND WIN32)
    configure_file(
        "${PROJECT_SOURCE_DIR}/../mimalloc/mi_build/Release/mimalloc.dll"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${BUILD_TYPE_DIR}/mimalloc.dll"
        COPYONLY
    )
    configure_file(
        "${PROJECT_SOURCE_DIR}/../mimalloc/mi_build/Release/mimalloc-redirect.dll"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${BUILD_TYPE_DIR}/mimalloc-redirect.dll"
        COPYONLY
    )
elseif (MIMALLOC_FOUND AND UNIX)
    configure_file(
        "${PROJECT_SOURCE_DIR}/../mimalloc/out/release/libmimalloc.so"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${BUILD_TYPE_DIR}/libmimalloc.so"
        COPYONLY
    )
endif()

set(all_files
    ../API/EbDebugMacros.h
    ../API/EbSvtAv1.h
    ../API/EbSvtAv1Enc.h
    ../API/EbSvtAv1ErrorCodes.h
    ../API/EbSvtAv1Formats.h
    ../API/EbSvtAv1Metadata.h
    app_config.c
    app_config.h
    app_context.c
    app_context.h
    app_input_y4m.c
    app_input_y4m.h
    app_main.c
    app_output_ivf.c
    app_output_ivf.h
    app_process_cmd.c
    svt_time.c
    svt_time.h
    )

# App Source Files
add_executable(SvtAv1EncApp
    ${all_files})

if(TARGET safestringlib)
    target_sources(SvtAv1EncApp PRIVATE $<TARGET_OBJECTS:safestringlib>)
endif()

#********** SET COMPILE FLAGS************
# Link the Encoder App
target_link_libraries(SvtAv1EncApp
        SvtAv1Enc
        ${MIMALLOC_LIB}
        ${LIBDOVI_LIBRARY}
        ${LIBHDR10PLUS_RS_LIBRARY})

target_include_directories(SvtAv1EncApp PRIVATE ${MIMALLOC_INCLUDE_DIR})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
    target_link_libraries(SvtAv1EncApp ${PLATFORM_LIBS})
elseif(UNIX)
    target_link_libraries(SvtAv1EncApp
        pthread
        m)
endif()

if(MSVC AND CMAKE_C_COMPILER_ID MATCHES "Clang")
  # ClangCL requires these flags for mimalloc's override
  target_compile_definitions(SvtAv1EncApp PRIVATE
    -D_WIN32_WINNT=0x0601   # Windows API version
    -D_CRT_SECURE_NO_WARNINGS
  )
  target_compile_options(SvtAv1EncApp PRIVATE /Zc:__cplusplus)
endif()


install(TARGETS SvtAv1EncApp RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})
