cmake_minimum_required(VERSION 3.5)
project(rtd_lite LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

option(RTDLITE_BUILD_CLI "Build rtdlite command-line executable" ON)

include(CheckCXXSourceCompiles)
include(GenerateExportHeader)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

# Shared library (core) - RTDLITE_LIBRARY excludes main()
add_library(rtd_lite SHARED RTDLite/rtdlite.cpp)
set_target_properties(rtd_lite PROPERTIES OUTPUT_NAME "rtd_lite" PREFIX "")
target_compile_definitions(rtd_lite PRIVATE RTDLITE_LIBRARY)
target_include_directories(rtd_lite
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/RTDLite>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    generate_export_header(rtd_lite)
endif()

# Optional CLI executable (standalone binary with main)
if(RTDLITE_BUILD_CLI)
    add_executable(rtdlite_cli RTDLite/rtdlite.cpp)
    target_include_directories(rtdlite_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/RTDLite)
endif()

# Install for standalone use
install(TARGETS rtd_lite
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES RTDLite/rtdlite.h DESTINATION include)