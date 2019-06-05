//
// Created by zhou on 2019-05-27.
//

#ifndef DEEPSTACK_CPP_TERMINAL_EQUITY_H
#define DEEPSTACK_CPP_TERMINAL_EQUITY_H


#include <torch/torch.h>
#include <ATen/ATen.h>
#include "../Settings/game_settings.h"
#include "../Settings/constants.h"
#include "../Game/board.h"
#include "../Game/card_tools.h"
#include "../runtime.h"

class TerminalEquity {

public:
    int river_boards_count = CardTools::get_boards_count(4);

    float (*equity_matrix_np)[hand_count] = new float[hand_count][hand_count];
    int (*sorted_hand_values_np)[hand_count] = new int[river_boards_count][hand_count];
    int (*ranks_np)[hand_count] = new int[river_boards_count][hand_count];
    float (*river_hand_abstract_np)[hand_count][max_abs_count] = new float[river_boards_count][hand_count][max_abs_count];
    float (*equity_matrix_next_street_np)[hand_count][hand_count] = new float[river_boards_count][hand_count][hand_count];
    float (*equity_matrix_next_street_np_)[max_abs_count] = new float[max_abs_count][max_abs_count];
    float (*fold_matrix_next_street_np)[max_abs_count] = new float[max_abs_count][max_abs_count];

    torch::Tensor preflop_equity_matrix;

    torch::Tensor equity_matrix = torch::zeros({hand_count, hand_count}, torch::kFloat32).to(device);

    torch::Tensor fold_matrix = torch::zeros({hand_count, hand_count}, torch::kFloat32).to(device);

    torch::Tensor equity_matrix_next_street;

    torch::Tensor fold_matrix_next_street;

    torch::Tensor river_hand_abstract;

    torch::Tensor mask_next_street;

    torch::Tensor possible_hand_indexes = torch::zeros(hand_count, torch::kFloat32).to(device);

    Board board;

    int river_hand_abstract_count = hand_count;

    TerminalEquity();

    ~TerminalEquity() = default;

    void _reset_tensors();

    void _set_river_abstract(Board &board);

    void _handle_blocking_cards();

    void _set_call_matrix();

    void _set_fold_matrix();

    void set_board(Board& src_board, bool force = false);

    void call_value(torch::Tensor &ranges, torch::Tensor &result);

    void call_value_next_street(torch::Tensor &ranges, torch::Tensor &result);

    void fold_value(torch::Tensor &ranges, torch::Tensor &result);

    void fold_value_next_street(torch::Tensor &ranges, torch::Tensor &result);
};


#endif //DEEPSTACK_CPP_TERMINAL_EQUITY_H
