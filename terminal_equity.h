//
// Created by zhou on 2019-05-27.
//

#ifndef DEEPSTACK_CPP_TERMINAL_EQUITY_H
#define DEEPSTACK_CPP_TERMINAL_EQUITY_H


#include <torch/torch.h>
#include <ATen/ATen.h>
#include "game_settings.h"
#include "constants.h"
#include "board.h"
#include "card_tools.h"
#include "runtime.h"

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


TerminalEquity::TerminalEquity() {
    Board b;
    this->set_board(b, true);
}

void TerminalEquity::_reset_tensors() {
    memset( equity_matrix_np, 0, hand_count * hand_count* sizeof(float) );
    memset( sorted_hand_values_np, 0, river_boards_count * hand_count * sizeof(int) );
    memset( ranks_np, 0, river_boards_count * hand_count * sizeof(int) );
    memset( river_hand_abstract_np, 0, river_boards_count * hand_count * max_abs_count * sizeof(float) );
    memset( equity_matrix_next_street_np, 0, river_boards_count * hand_count * hand_count * sizeof(float) );
    memset( equity_matrix_next_street_np_, 0, max_abs_count * max_abs_count * sizeof(float) );
    memset( fold_matrix_next_street_np, 0, max_abs_count * max_abs_count * sizeof(float) );

    equity_matrix.zero_();
}

void TerminalEquity::_set_river_abstract(Board &b) {

    river_hand_abstract_count = set_river_abstract_combine_cpp(b.cards, river_hand_abstract_np, equity_matrix_next_street_np_, fold_matrix_next_street_np);

    river_hand_abstract = torch::from_blob(river_hand_abstract_np, {river_boards_count, hand_count, max_abs_count}, torch::kFloat32).slice(2, 0, river_hand_abstract_count).to(device);

    equity_matrix_next_street = torch::from_blob(equity_matrix_next_street_np_, {max_abs_count, max_abs_count}, torch::kFloat32).slice(0, 0, river_hand_abstract_count).slice(1, 0, river_hand_abstract_count).to(device);

    fold_matrix_next_street = torch::from_blob(fold_matrix_next_street_np, {max_abs_count, max_abs_count}, torch::kFloat32).slice(0, 0, river_hand_abstract_count).slice(1, 0, river_hand_abstract_count).to(device);

    mask_next_street = river_hand_abstract.sum(2).view({boards_count[4], 1, -1}).expand({-1, 2, -1});
    mask_next_street.masked_fill_(mask_next_street > 0, 1);
    mask_next_street = mask_next_street.toType(torch::kByte);;
    mask_next_street *= -1;
    mask_next_street += 1;
}

void TerminalEquity::_handle_blocking_cards() {
    CardTools::get_possible_hand_indexes(board, possible_hand_indexes);
    torch::Tensor possible_hand_matrix = possible_hand_indexes.expand_as(fold_matrix);
    fold_matrix *= possible_hand_matrix;
    fold_matrix *= possible_hand_matrix.t();
}

void TerminalEquity::_set_call_matrix() {

    switch (board.street()) {

        case 1: {
            equity_matrix.copy_(preflop_equity_matrix);
            break;
        }
        case 2: {
            int card_0 = *min_element(board.cards, board.cards+2);
            int card_2 = *max_element(board.cards, board.cards+2);
            int card_1 = board.cards[0] + board.cards[1] + board.cards[2] - card_0 - card_2;
            char equity_matrix_file[50];
            sprintf(equity_matrix_file, "%s%d_%d_%d.bin", flop_equity_matrix_dir, card_0, card_1, card_2);
            read_pointer((float*)equity_matrix_np, equity_matrix_file);
            equity_matrix.set_data(torch::from_blob(equity_matrix_np, equity_matrix.sizes(), equity_matrix.dtype()));
            break;
        }
        case 3: {
            set_call_matrix_street_3_cpp(board.cards, equity_matrix_np, equity_matrix_next_street_np, sorted_hand_values_np, ranks_np);
            equity_matrix.set_data(
                    torch::from_blob(equity_matrix_next_street_np, {river_boards_count, hand_count, hand_count}, torch::kFloat32).sum(0).to(device) / 44);
            break;
        }
        case 4: {
            set_call_matrix_4_cpp(board.cards, equity_matrix_np);
            equity_matrix.set_data(torch::from_blob(equity_matrix_np, equity_matrix.sizes(), equity_matrix.dtype()));
            break;
        }
        default:
            break;
    }
}

void TerminalEquity::_set_fold_matrix() {
    fold_matrix.copy_(CardTools::hand_collide);
    _handle_blocking_cards();
}


void TerminalEquity::set_board(Board &src_board, bool force) {
    if (board.is_same(src_board) && ! force)
        return;
    board.copy_(src_board);
    this->_reset_tensors();

    if (board.street() == 4)
        this->_set_river_abstract(board);
    else
        this->river_hand_abstract_count = hand_count;
    this->_set_call_matrix();
    this->_set_fold_matrix();
}


void TerminalEquity::call_value(torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, equity_matrix);
}

void TerminalEquity::call_value_next_street(torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, equity_matrix_next_street);
}

void TerminalEquity::fold_value(torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, fold_matrix);
}

void TerminalEquity::fold_value_next_street(torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, fold_matrix_next_street);
}


#endif //DEEPSTACK_CPP_TERMINAL_EQUITY_H
