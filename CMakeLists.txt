cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

project(deepstack_cpp)

set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED)

include_directories(/usr/local/include/ATen /usr/local/include/c10 /usr/local/include/torch)
link_directories(/usr/local/lib)

add_executable(deepstack_cpp main.cpp game_settings.h terminal_equity.h board.h board.h card_tools.h constants.h io.h runtime.h)
target_link_libraries(deepstack_cpp "${TORCH_LIBRARIES}")
#set_property(TARGET deepstack_cpp PROPERTY CXX_STANDARD 11)
