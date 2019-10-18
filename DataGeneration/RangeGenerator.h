//
// Created by zhou on 19-9-24.
//

#ifndef DEEPSTACK_CPP_RANGEGENERATOR_H
#define DEEPSTACK_CPP_RANGEGENERATOR_H


#include <vector>
#include <torch/torch.h>
#include "../runtime.h"
#include "../Game/board.h"
#include "../Game/card_tools.h"
#include "../Settings/constants.h"
#include "../Settings/game_settings.h"

class RangeGenerator {

public:

    std::function<float()> dice = std::bind(distribution, generator);
    int possible_hands_mask[hand_count];
    int possible_hands_count;
    std::vector<pair<int, int>> hand_strengths;

    void set_board(const Board& board);
    void generate(torch::Tensor& ranges);
    void _generate_sorted_range(float* sorted_ranges);
    void _generate_recursion(float* sorted_ranges, float mass, int start, int end);
    void generate_board(int street, Board& board);
    int generate_pot();
};


#endif //DEEPSTACK_CPP_RANGEGENERATOR_H
