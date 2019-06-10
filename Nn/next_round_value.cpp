//
// Created by zhou on 19-6-10.
//


#include "next_round_value.h"


NextRoundValue::NextRoundValue(const int _street, ValueNn *src_value_nn, ValueNn *src_aux_nn) {
    value_nn = src_value_nn;
    aux_value_nn = src_value_nn;
    street = _street;
    board_count = CardTools::get_boards_count(street+1);
    _range_matrix = torch::zeros({hand_count, board_count * bucket_count}, torch::kFloat32).to(device);
    _range_matrix_board_view = _range_matrix.view({hand_count, board_count, bucket_count});
    if (street == 1)
        init_bucketing();
    else {
        read_pointer(canonical_map_turn, canonical_map_turn_file);
        read_pointer(assignments_turn, assignments_turn_file);
    }
}

void NextRoundValue::init_bucketing(Board *board_ptr) {
    if (street == 1) {
        auto p = new int32_t[board_count * hand_count];
        read_pointer(p, board_buckets_file);
        board_buckets = torch::from_blob(p, {board_count, hand_count}, torch::kInt32).to(device) * -1;
        delete[] p;

        impossible_mask = torch::lt(board_buckets, 0);
        board_indexes = board_buckets.clone().toType(torch::kLong);
        board_indexes.masked_fill_(impossible_mask, 0);
        board_indexes_scatter = board_buckets.clone().toType(torch::kLong);
        board_indexes_scatter.masked_fill_(impossible_mask, bucket_count);

        _aux_range_matrix = torch::zeros({hand_count, aux_bucket_count}, torch::kFloat32).to(device);
        int h_idx = -1, bucket_id;
        for (int card_0 = 0; card_0 < card_count; ++card_0)
            for (int card_1=card_0 + 1; card_1 < card_count; ++card_1) {
                h_idx ++;
                bucket_id = street_1_abstraction[h_idx];
                _aux_range_matrix[h_idx][bucket_id] = 1;
            }
        _aux_reverse_value_matrix = _aux_range_matrix.transpose(0,1).clone();
        weight_constant = 6.0 / ((card_count - 4) * (card_count - 5) * (card_count - 6));
    }
    else {
        int card_0 = *min_element(board_ptr->cards, board_ptr->cards+3);
        int card_2 = *max_element(board_ptr->cards, board_ptr->cards+3);
        int card_1 = board_ptr->cards[0] + board_ptr->cards[1] + board_ptr->cards[2] - card_0 - card_2;
        char range_matrix_cache_file[50];
        sprintf(range_matrix_cache_file, "%s%d_%d_%d.bin", range_matrix_cache_root_file, card_0, card_1, card_2);
        if (access(range_matrix_cache_file, 0) != 0) {
            auto p = new uint8_t[hand_count * board_count * bucket_count];
            read_pointer(p, range_matrix_cache_file);
            _range_matrix_board_view = torch::from_blob(p, {hand_count, board_count, bucket_count}, torch::kUInt8).to(device).to(torch::kFloat32);
            _range_matrix = _range_matrix_board_view.view({hand_count, board_count * bucket_count});
        }
        else {
            _range_matrix = torch::zeros({hand_count, board_count * bucket_count}, torch::kFloat32).to(device);
            _range_matrix_board_view = _range_matrix.view({hand_count, board_count, bucket_count});
            auto boards_ptr = new Board[board_count];
            CardTools::get_x_round_boards(*board_ptr, boards_ptr);
            for (int idx=0; idx<board_count; ++idx) {
                Board& board = boards_ptr[idx];
                auto possible_hands = new int[hand_count];
                CardTools::get_possible_hand_indexes(board, possible_hands);
                int h_idx = -1;
                int table_hand[6];
                memcpy(table_hand, board.cards, 4 * sizeof(int));
                for (int _card_0 = 0; _card_0 < card_count; ++_card_0) {
                    for (int _card_1 = _card_0 + 1; _card_1 < card_count; ++_card_1) {
                        h_idx ++;
                        if (possible_hands[h_idx] == 0)
                            continue;
                        table_hand[4] = _card_0;
                        table_hand[5] = _card_1;
                        int all_state_id_4 = get_state_id_in_4(table_hand);
                        _range_matrix_board_view[h_idx][idx][assignments_turn[canonical_map_turn[all_state_id_4]]] = 1;
                    }
                }

            }
        }
        _reverse_value_matrix = _range_matrix.transpose(0, 1).clone();
        weight_constant = 1.0 / (board_count - 4);
        _reverse_value_matrix *= weight_constant;
    }
}

void NextRoundValue::_hand_range_to_bucket_range(torch::Tensor& hand_range, torch::Tensor& bucket_range) {
    if (street == 1) {
        torch::Tensor other_bucket_range = bucket_range.view({-1, board_count, bucket_count + 1});
        other_bucket_range.zero_();
        torch::Tensor indexes = board_indexes_scatter.view({1, board_count, hand_count}).expand({bucket_range.sizes()[0], -1, -1});
        other_bucket_range.scatter_add_(2, indexes, hand_range.view({-1, 1, hand_count}).expand({hand_range.sizes()[0], board_count, -1}));
    }
    else
        bucket_range.copy_(torch::matmul(hand_range, _range_matrix));
}

void NextRoundValue::_bucket_value_to_hand_value(torch::Tensor& bucket_value, torch::Tensor& hand_value) {
    if (street == 1) {
        torch::Tensor indexes = board_indexes.view({1, board_count, hand_count}).expand({bucket_value.sizes()[0], -1, -1});
        values_per_board.copy_(torch::gather(bucket_value.view({bucket_value.sizes()[0], board_count, bucket_count}), 2, indexes));
        torch::Tensor impossible = impossible_mask.view({1, board_count, hand_count}).expand({bucket_value.sizes()[0], -1, -1});
        values_per_board.masked_fill_(impossible, 0);
        hand_value.copy_(values_per_board.sum(1));
        hand_value.mul_(weight_constant);
    }
    else
        hand_value.copy_(torch::matmul(bucket_value, _reverse_value_matrix));
}

void NextRoundValue::_hand_range_to_bucket_range_on_board(const int board_idx, torch::Tensor& hand_range, torch::Tensor& bucket_range) {
    torch::Tensor other_bucket_range = bucket_range.view({-1, bucket_count + 1});
    other_bucket_range.zero_();
    torch::Tensor indexes = board_indexes_scatter.view({1, board_count, hand_count}).slice(
            1, board_idx, board_idx+1, 1).squeeze().expand({bucket_range.sizes()[0], -1});
    other_bucket_range.scatter_add_(1, indexes, hand_range.view({-1, hand_count}).expand({hand_range.sizes()[0], -1}));
}

void NextRoundValue::_bucket_value_to_hand_value_on_board(Board& board, torch::Tensor& bucket_value, torch::Tensor& hand_value) {
    if (street == 1) {
        int board_idx = CardTools::get_board_index(board);
        torch::Tensor indexes = board_indexes.view({1, board_count, hand_count}).slice(
                1, board_idx, board_idx+1, 1).squeeze().expand({bucket_value.sizes()[0], -1});
        aux_values_per_board.copy_(torch::gather(bucket_value.view({bucket_value.sizes()[0], bucket_count}), 1, indexes));
        torch::Tensor impossible = impossible_mask.view({1, board_count, hand_count}).slice(
                1, board_idx, board_idx + 1, 1).squeeze().expand({bucket_value.sizes()[0], -1});
        aux_values_per_board.masked_fill_(impossible, 0);
        hand_value.copy_(aux_values_per_board);
    }
    else {
        int board_idx = CardTools::get_board_index(board);
        torch::Tensor board_matrix = _range_matrix_board_view.slice(1, board_idx, board_idx + 1, 1).squeeze().transpose(0, 1);
        torch::Tensor serialized_hand_value = hand_value.view({-1, hand_count});
        torch::Tensor serialized_bucket_value = bucket_value.slice(2, board_idx, board_idx + 1, 1).squeeze().view({-1, bucket_count});
        serialized_hand_value.copy_(torch::matmul(serialized_bucket_value, board_matrix));
    }
}

void NextRoundValue::start_computation(torch::Tensor& src_pot_sizes) {
    iter = -1;
    pot_sizes = src_pot_sizes.view({-1, 1}).clone();
    batch_size = pot_sizes.sizes()[0];
    _values_are_prepared = false;
}

void NextRoundValue::init_var() {

}














NextRoundValue& get_flop_value() {
    static NextRoundValue flop_value(1, &get_turn_nn());
    return flop_value;
}

NextRoundValue& get_turn_value() {
    static NextRoundValue turn_value(2, &get_flop_nn(), &get_aux_nn());
    return turn_value;
}
