cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(deepstack_cpp)

find_package(Torch REQUIRED)

set(Boost_INCLUDE_DIR /home/zhou/Downloads/boost-1.58/include/boost-1_58)
set(Boost_LIBRARY_DIR /home/zhou/Downloads/boost-1.58/lib)
#include_directories(${Boost_INCLUDE_DIR})
#link_directories(${Boost_LIBRARY_DIR})


#set(Boost_LIBRARY_DIR /home/zhou/Downloads/boost_lib)
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)
find_package(Boost 1.58.0 REQUIRED COMPONENTS system thread filesystem)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})

#include_directories(/usr/local/include)
#link_directories(/usr/local/lib)

add_executable(deepstack_cpp main.cpp Settings/game_settings.h TerminalEquity/terminal_equity.h Game/board.h Game/board.h Game/card_tools.h
        Settings/constants.h io.h runtime.h Tree/poker_tree_builder.h Game/bet_sizing.h Tree/node.h Lookahead/lookahead.h Lookahead/lookahead.cpp
        Tree/poker_tree_builder.cpp Game/bet_sizing.cpp TerminalEquity/terminal_equity.cpp runtime.cpp Lookahead/LookaheadBuilder.cpp
        Game/card_tools.cpp Lookahead/resolving.h Lookahead/resolving.cpp Nn/next_round_value.h Nn/next_round_value.cpp Nn/value_nn.h Nn/value_nn.cpp
        Lookahead/cfrd_gadget.h Lookahead/cfrd_gadget.cpp Web/server_http.hpp Web/status_code.hpp Web/utility.hpp Player/continual_resolving.cpp
        Player/continual_resolving.h Game/card_to_string_conversion.h Game/card_to_string_conversion.cpp Settings/game_settings.cpp Settings/constants.cpp
        DataGeneration/RangeGenerator.h DataGeneration/RangeGenerator.cpp DataGeneration/turn_generator.h DataGeneration/turn_generator.cpp)
target_link_libraries(deepstack_cpp PUBLIC Boost::system Boost::thread Boost::filesystem ${HYRISE_SQLPARSER} "${TORCH_LIBRARIES}")

