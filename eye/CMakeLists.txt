# CMakeList.txt : CMake project for Subproject, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18)

project (eye VERSION ${realsense_VERSION})

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
    "${REALSENSE_SDK_ROOT_DIR}/include"
    "${REALSENSE_SDK_ROOT_DIR}/samples"
    "${OPENCV3412_ROOT_DIR}/include"
)

target_link_libraries (${PROJECT_NAME}
    "${REALSENSE_SDK_ROOT_DIR}/lib/x64/realsense2.lib"
)
target_link_libraries (${PROJECT_NAME} optimized
    ${OPENCV3412_ROOT_DIR}/x64/vc15/lib/opencv*3412.lib
)
target_link_libraries (${PROJECT_NAME} debug
    ${OPENCV3412_ROOT_DIR}/x64/vc15/lib/opencv*3412d.lib
)

if (WIN32)
    target_compile_definitions (${PROJECT_NAME}
    PUBLIC
        _CRT_SECURE_NO_WARNINGS
        ${PROJECT_NAME}_DLLEXPORT
    )
endif()

# Copy RealSense SDK dll
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${REALSENSE_SDK_ROOT_DIR}/bin/x64/realsense2.dll" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E echo "Copy realsense2.dll to ${PROJECT_NAME} exe directory"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV3412_ROOT_DIR}/x64/vc15/bin/opencv_world3412.dll" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV3412_ROOT_DIR}/x64/vc15/bin/opencv_world3412d.dll" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E echo "Copy opencv dll to ${PROJECT_NAME} exe directory"
)

# Define install patterns (library project need install target)
# Install binaries
install (
    TARGETS ${PROJECT_NAME}
    CONFIGURATIONS Debug Release
    RUNTIME DESTINATION "bin"
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_EXECUTE WORLD_READ
)
