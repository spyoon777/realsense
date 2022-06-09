# CMakeList.txt : CMake project for Subproject, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18)

project (eye VERSION 0.1.0.0 LANGUAGES CXX)

# configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
#                 "${CMAKE_CURRENT_SOURCE_DIR}/config.h" @ONLY)

add_executable (${PROJECT_NAME})
target_sources (${PROJECT_NAME}
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)

target_include_directories (${PROJECT_NAME}
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries (${PROJECT_NAME}
)

if (WIN32)
    target_compile_definitions (${PROJECT_NAME}
    PUBLIC
        _CRT_SECURE_NO_WARNINGS
        EYE_DLLEXPORT
    )
endif()

# Define install patterns (library project need install target)
# Install binaries
install (
    TARGETS ${PROJECT_NAME}
    CONFIGURATIONS Debug Release
    RUNTIME DESTINATION "bin"
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_EXECUTE WORLD_READ
)
