cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(deepstack_cpp)

find_package(Torch REQUIRED)

include_directories(/usr/local/include/ATen /usr/local/include/c10 /usr/local/include/torch)
link_directories(/usr/local/lib)

add_executable(deepstack_cpp main.cpp Settings/game_settings.h TerminalEquity/terminal_equity.h Game/board.h Game/board.h Game/card_tools.h Settings/constants.h io.h runtime.h Tree/poker_tree_builder.h Game/bet_sizing.h Tree/node.h Lookahead/lookahead.h Lookahead/lookahead.cpp Tree/poker_tree_builder.cpp Game/bet_sizing.cpp TerminalEquity/terminal_equity.cpp runtime.cpp Lookahead/LookaheadBuilder.cpp Game/card_tools.cpp Lookahead/resolving.h Lookahead/resolving.cpp)
target_link_libraries(deepstack_cpp "${TORCH_LIBRARIES}")

