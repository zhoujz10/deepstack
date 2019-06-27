//
// Created by zhou on 19-6-5.
//

#include "terminal_equity.h"


TerminalEquity::TerminalEquity() {
    auto p = new float[hand_count * hand_count];
    read_pointer(p, preflop_equity_matrix_file);
    preflop_equity_matrix.copy_(torch::from_blob(p, {hand_count, hand_count}, torch::kFloat32));
    delete[] p;
    this->set_board(Board(), true);
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
    card_tools.get_possible_hand_indexes(board, possible_hand_indexes);
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
            int card_0 = *min_element(board.cards, board.cards+3);
            int card_2 = *max_element(board.cards, board.cards+3);
            int card_1 = board.cards[0] + board.cards[1] + board.cards[2] - card_0 - card_2;
            char equity_matrix_file[50];
            sprintf(equity_matrix_file, "%s%d_%d_%d.bin", flop_equity_matrix_dir, card_0, card_1, card_2);
            if (access(equity_matrix_file, 0) == 0) {
                read_pointer((float *) equity_matrix_np, equity_matrix_file);
                std::cout << "equity matrix read." << std::endl;
            }
            else
                set_call_matrix_cpp(board.cards, equity_matrix_np);
            equity_matrix.copy_(torch::from_blob(equity_matrix_np, equity_matrix.sizes(), equity_matrix.dtype()));
            break;
        }
        case 3: {
            set_call_matrix_street_3_cpp(board.cards, equity_matrix_np, equity_matrix_next_street_np, sorted_hand_values_np, ranks_np);
            equity_matrix.copy_(
                    torch::from_blob(equity_matrix_next_street_np, {river_boards_count, hand_count, hand_count}, torch::kFloat32).sum(0).to(device) / 44);
            break;
        }
        case 4: {
            set_call_matrix_4_cpp(board.cards, equity_matrix_np);
            equity_matrix.copy_(torch::from_blob(equity_matrix_np, equity_matrix.sizes(), equity_matrix.dtype()));
            break;
        }
        default:
            break;
    }
}

void TerminalEquity::_set_fold_matrix() {
    fold_matrix.copy_(card_tools.hand_collide);
    _handle_blocking_cards();
}


void TerminalEquity::set_board(const Board &src_board, bool force) {
    if (board.is_same(src_board) && ! force)
        return;
    board.copy_(src_board);
    this->_reset_tensors();

    if (board.street() == 3)
        this->_set_river_abstract(board);
    else
        this->river_hand_abstract_count = hand_count;
    this->_set_call_matrix();
    this->_set_fold_matrix();
}


void TerminalEquity::call_value(const torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, equity_matrix);
}

void TerminalEquity::call_value_next_street(const torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, equity_matrix_next_street);
}

void TerminalEquity::fold_value(const torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, fold_matrix);
}

void TerminalEquity::fold_value_next_street(const torch::Tensor &ranges, torch::Tensor &result) {
    torch::matmul_out(result, ranges, fold_matrix_next_street);
}


TerminalEquity& get_terminal_equity() {
    static TerminalEquity terminal_equity;
    return terminal_equity;
}
