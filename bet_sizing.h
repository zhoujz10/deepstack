//
// Created by zhou on 19-6-2.
//

#ifndef DEEPSTACK_CPP_BET_SIZING_H
#define DEEPSTACK_CPP_BET_SIZING_H


#include <vector>
#include <array>
#include <tuple>
#include "node.h"
#include "constants.h"


void get_possible_bets(Node& node, int street, int depth, bool is_next,
        std::vector<std::array<int, 2>>& possible_bets, std::vector<float>& used_pot_fractions);

#endif //DEEPSTACK_CPP_BET_SIZING_H
