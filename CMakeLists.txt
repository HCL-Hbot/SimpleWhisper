# Minimum version should be low enough that everyone is able to use it
cmake_minimum_required(VERSION 3.14)

project(simplewhisper CXX)

# Needed for fetching model & GGML
include(FetchContent)

# Some of the more popular gpu options (VULKAN is great for NON-nvidia & CUDA is almost must-have for nvidia)
option(GGML_VULKAN "Run with vulkan backend" OFF)
option(GGML_CUDA "Run with CUDA backend" OFF)

# Perf options enables the profiling (tok/s) of the runs
option(GGML_PERF ON)

# This option (build_examples) gets disables automatically when imported
option(SIMPLEWHISPER_BUILD_EXAMPLE "Build example" ON)

# Optionally enable ROS2 package build (Requires ROS2 jazzy to be installed!)
option(SIMPLEWHISPER_ROS2_PACKAGE_BUILD "Build a ROS2 package" OFF)

find_package(SDL2 REQUIRED)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(SIMPLEWHISPER_BUILD_EXAMPLE OFF)
endif()

# Get GGML if not present already
if(TARGET ggml)
else()
  FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG f3a375f20bf56860b30e7c511d03593a1e393345
  )
  set(GGML_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GGML_RUNTIME_DIR})

  FetchContent_MakeAvailable(ggml)
endif()

FetchContent_Declare(
   whisper
   GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
   GIT_TAG 0778b6ff5fb86439a4aeaec32359416413ecaa2d
 )

FetchContent_MakeAvailable(whisper)
# Add simplewhisper wrapper
add_library(simplewhisper ${CMAKE_CURRENT_LIST_DIR}/src/simplewhisper.cpp)
target_include_directories(simplewhisper PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(simplewhisper whisper ggml ${SDL2_LIBRARIES})

if(SIMPLEWHISPER_ROS2_PACKAGE_BUILD)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  find_package(audio_tools REQUIRED)
  add_executable(simplewhisper_node ${CMAKE_CURRENT_LIST_DIR}/binding/ros2/whisper_node.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/binding/ros2/audio_async_ros/audio_async_ros.cpp)
  ament_target_dependencies(simplewhisper_node
    rclcpp
    std_msgs
    std_srvs
    audio_tools
  )
  target_link_libraries(simplewhisper_node simplewhisper)
  install(TARGETS
    simplewhisper_node
    DESTINATION lib/${PROJECT_NAME})
    install(TARGETS
      simplewhisper whisper
      DESTINATION lib/
    )
    install(TARGETS
      simplewhisper whisper
      DESTINATION lib64/
    )
endif()

if(SIMPLEWHISPER_BUILD_EXAMPLE)
  if(SIMPLEWHISPER_ROS2_PACKAGE_BUILD)
  else()
    add_executable(simple_speech_demo ${CMAKE_CURRENT_LIST_DIR}/example/simple_speech_demo/simple_speech_demo.cpp
                                      ${CMAKE_CURRENT_LIST_DIR}/example/simple_speech_demo/audio_async/audio_async.cpp)
    target_include_directories(simple_speech_demo PRIVATE ${CMAKE_CURRENT_LIST_DIR}/example/simple_speech_demo/audio_async)
    target_link_libraries(simple_speech_demo simplewhisper -lgomp)
  endif()
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
  endif()

endif()
if(SIMPLEWHISPER_ROS2_PACKAGE_BUILD)
  ament_package()
endif()
