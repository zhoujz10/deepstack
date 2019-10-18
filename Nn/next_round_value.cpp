//
// Created by zhou on 19-6-10.
//


#include "next_round_value.h"


NextRoundValue::NextRoundValue(const int _street, ValueNn *src_value_nn, ValueNn *src_aux_nn) {
    value_nn = src_value_nn;
    aux_value_nn = src_aux_nn;
    street = _street;
    board_count = CardTools::get_boards_count(street+1);
    if (street == 1)
        init_bucketing();
    else {
        _range_matrix = torch::zeros({hand_count, board_count * bucket_count}, torch::kFloat32).to(device);
        _range_matrix_board_view = _range_matrix.view({hand_count, board_count, bucket_count});
        canonical_map_turn = new uint32_t[305377800];
        assignments_turn = new uint16_t[13960050];
        read_pointer(canonical_map_turn, canonical_map_turn_file);
        read_pointer(assignments_turn, assignments_turn_file);
    }
}

void NextRoundValue::init_bucketing(Board *board_ptr) {
    if (street == 1) {
        auto p = new int32_t[board_count * hand_count];
        read_pointer(p, board_buckets_file);
        board_buckets = torch::from_blob(p, {board_count, hand_count}, torch::kInt32).to(device);
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
        sprintf(range_matrix_cache_file, "%s%d_%d_%d.npy", range_matrix_cache_root_file, card_0, card_1, card_2);
        if (access(range_matrix_cache_file, 0) == 0) {
            unsigned npy_shift = 128;
            auto p = new uint8_t[hand_count * board_count * bucket_count + npy_shift];
            ifstream f_read(range_matrix_cache_file, ios::binary);
            f_read.read( reinterpret_cast<char *>(&p[0]), (uint64_t)hand_count * board_count * bucket_count * sizeof(uint8_t) + npy_shift );
            _range_matrix_board_view = torch::from_blob(p+npy_shift, {hand_count, board_count, bucket_count}, torch::kUInt8).to(device).to(torch::kFloat32);
            _range_matrix = _range_matrix_board_view.view({hand_count, board_count * bucket_count});

//            char temp_file[50];
//            sprintf(temp_file, "temp_%d_%d_%d.npy", card_0, card_1, card_2);
//            write_pointer(p+npy_shift, temp_file, (uint32_t)hand_count * board_count * bucket_count);

            delete[] p;
            std::cout << "range matrix cache file read." << std::endl;
        }
//        char range_matrix_cache_small_file[50];
//        sprintf(range_matrix_cache_small_file, "%s%d_%d_%d.npy", range_matrix_cache_small_root_file, card_0, card_1, card_2);
//        if (access(range_matrix_cache_small_file, 0) == 0) {
//            unsigned npy_shift = 128;
//            auto p_idx = new uint16_t[1128 * 49 * 3 + npy_shift / 2];
//            ifstream f_read(range_matrix_cache_small_file, ios::binary);
//            f_read.read( reinterpret_cast<char *>(&p_idx[0]), 1128 * 49 * 3 * sizeof(uint16_t) + npy_shift );
//            _range_matrix_board_view.zero_();
//            for (int i = 0; i < 1128 * 49; ++i) {
//                _range_matrix_board_view[p_idx[i*3+npy_shift/2]][p_idx[i*3+1+npy_shift/2]][p_idx[i*3+2+npy_shift/2]] = 1;
//            }
//            delete[] p_idx;
//            std::cout << "range matrix cache file read." << std::endl;
//        }
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

void NextRoundValue::_hand_range_to_bucket_range(torch::Tensor& hand_range, torch::Tensor& _bucket_range) {
    if (street == 1) {
        torch::Tensor other_bucket_range = _bucket_range.view({-1, board_count, bucket_count + 1});
        other_bucket_range.zero_();
        torch::Tensor indexes = board_indexes_scatter.view({1, board_count, hand_count}).expand({bucket_range.sizes()[0], -1, -1});
        other_bucket_range.scatter_add_(2, indexes, hand_range.view({-1, 1, hand_count}).expand({hand_range.sizes()[0], board_count, -1}));
    }
    else
        _bucket_range.copy_(torch::matmul(hand_range, _range_matrix));
}

void NextRoundValue::_hand_range_to_bucket_range_last_20(torch::Tensor& hand_range, torch::Tensor& _bucket_range) {
    torch::Tensor other_bucket_range = _bucket_range.view({-1, board_count, bucket_count + 1});
    other_bucket_range.zero_();
    torch::Tensor indexes = board_indexes_scatter.view({1, board_count, hand_count}).expand({_bucket_range.sizes()[0], -1, -1});
    other_bucket_range.scatter_add_(2, indexes, hand_range.view({-1, 1, hand_count}).expand({hand_range.sizes()[0], board_count, -1}));
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

void NextRoundValue::_bucket_value_to_hand_value_last_20(torch::Tensor& bucket_value, torch::Tensor& hand_value) {
    torch::Tensor indexes = board_indexes.view({1, board_count, hand_count}).expand({bucket_value.sizes()[0], -1, -1});
    values_per_board.copy_(torch::gather(bucket_value.view({bucket_value.sizes()[0], board_count, bucket_count}), 2, indexes));
    torch::Tensor impossible = impossible_mask.view({1, board_count, hand_count}).expand({bucket_value.sizes()[0], -1, -1});
    values_per_board.masked_fill_(impossible, 0);
    hand_value.copy_(values_per_board.sum(1));
    hand_value.mul_(weight_constant);
}

void NextRoundValue::_hand_range_to_bucket_range_on_board(const int board_idx, torch::Tensor& hand_range, torch::Tensor& _bucket_range) {
    torch::Tensor other_bucket_range = _bucket_range.view({-1, bucket_count + 1});
    other_bucket_range.zero_();
    torch::Tensor indexes = board_indexes_scatter.view({1, board_count, hand_count}).slice(
            1, board_idx, board_idx+1, 1).squeeze().expand({_bucket_range.sizes()[0], -1});
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

void NextRoundValue::_hand_range_to_bucket_range_aux(const torch::Tensor& hand_range, torch::Tensor& _bucket_range) {
    _bucket_range.copy_(torch::matmul(hand_range, _aux_range_matrix));
}

void NextRoundValue::_bucket_value_to_hand_value_aux(const torch::Tensor& bucket_value, torch::Tensor& hand_value) {
    hand_value.copy_(torch::matmul(bucket_value, _aux_reverse_value_matrix));
}

void NextRoundValue::start_computation(torch::Tensor& src_pot_sizes) {
    iter = -1;
    pot_sizes = src_pot_sizes.view({-1, 1}).clone();
    batch_size = pot_sizes.sizes()[0];
    _values_are_prepared = false;
}

void NextRoundValue::init_var() {
    aux_next_round_inputs = torch::zeros({batch_size, (aux_bucket_count * constants.players_count + 1)}, torch::kFloat32).to(device);
    aux_next_round_values = torch::zeros({batch_size, constants.players_count, aux_bucket_count}, torch::kFloat32).to(device);
    aux_next_round_extended_range = torch::zeros({batch_size, constants.players_count, aux_bucket_count}, torch::kFloat32).to(device);
    aux_next_round_serialized_range = aux_next_round_extended_range.view({-1, aux_bucket_count});
    aux_values_per_board = torch::zeros({batch_size * constants.players_count, hand_count}, torch::kFloat32).to(device);
    aux_value_normalization = torch::zeros({batch_size, constants.players_count}, torch::kFloat32).to(device);
    aux_next_round_inputs.slice(1, aux_bucket_count * constants.players_count, max_size, 1).squeeze(1).copy_(pot_sizes.view(-1) / params::stack);
}

void NextRoundValue::get_value_aux(torch::Tensor& ranges, torch::Tensor& values, const int next_board_idx) {

    assert (ranges.sizes()[0] == batch_size);
    iter ++;
//    if (iter == 0) {
////        initializing data structures
//        aux_next_round_inputs = torch::zeros({batch_size, (aux_bucket_count * constants.players_count + 1)}, torch::kFloat32).to(device);
//        aux_next_round_values = torch::zeros({batch_size, constants.players_count, aux_bucket_count}, torch::kFloat32).to(device);
//        aux_next_round_extended_range = torch::zeros({batch_size, constants.players_count, aux_bucket_count}, torch::kFloat32).to(device);
//        aux_next_round_serialized_range = aux_next_round_extended_range.view({-1, aux_bucket_count});
//        aux_values_per_board = torch::zeros({batch_size * constants.players_count, hand_count}, torch::kFloat32).to(device);
//        aux_value_normalization = torch::zeros({batch_size, constants.players_count}, torch::kFloat32).to(device);
////        handling pot feature for the nn
//        aux_next_round_inputs.slice(1, aux_bucket_count * constants.players_count, max_size, 1).squeeze().copy_(pot_sizes.view(-1) / stack);
//    }

    if (iter == 0)
        init_var();

    bool use_memory = (iter >= cfr_skip_iters[street]) && next_board_idx > -1;

    if (use_memory && (iter == cfr_skip_iters[street])) {
//        first iter that we need to remember something - we need to init data structures
        bucket_range = torch::zeros({batch_size * constants.players_count, bucket_count}, torch::kFloat32).to(device);
        next_round_inputs = torch::zeros({batch_size, (bucket_count * constants.players_count + 1)}, torch::kFloat32).to(device);
        next_round_values = torch::zeros({batch_size, constants.players_count, bucket_count}, torch::kFloat32).to(device);
        next_round_extended_range = torch::zeros({batch_size, constants.players_count, bucket_count+1}, torch::kFloat32).to(device);
        next_round_serialized_range = next_round_extended_range.view({-1, bucket_count+1});

        value_normalization = torch::zeros({batch_size, constants.players_count}, torch::kFloat32).to(device);

        range_normalization_memory = torch::zeros({batch_size * constants.players_count, 1}, torch::kFloat32).to(device);
        counterfactual_value_memory = torch::zeros({batch_size, constants.players_count, bucket_count}, torch::kFloat32).to(device);

        next_round_inputs.slice(1, aux_bucket_count * constants.players_count, max_size, 1).copy_(
                aux_next_round_inputs.slice(1, aux_bucket_count * constants.players_count, max_size, 1));
    }

    torch::Tensor _ranges = ranges.view({batch_size * constants.players_count, -1});
    torch::Tensor _aux_next_round_extended_range = aux_next_round_extended_range.view({batch_size * constants.players_count, -1});

    _hand_range_to_bucket_range_aux(_ranges, _aux_next_round_extended_range);
    aux_range_normalization = aux_next_round_serialized_range.sum(1);
    torch::Tensor rn_view = aux_range_normalization.view({batch_size, constants.players_count});
    for (int player=0; player<constants.players_count; ++player)
        aux_value_normalization.slice(1, player, player+1, 1).copy_(rn_view.slice(1, 1-player, 2-player, 1));

    if (use_memory) {
        torch::Tensor _next_round_extended_range = next_round_extended_range.view({batch_size * constants.players_count, -1});
        _hand_range_to_bucket_range_on_board(next_board_idx, _ranges, _next_round_extended_range);
        range_normalization = next_round_serialized_range.slice(1, 0, bucket_count, 1).sum(1);
        torch::Tensor rnb_view = range_normalization.view({batch_size, constants.players_count});
        for (int player=0; player<constants.players_count; ++player)
            value_normalization.slice(1, player, player+1, 1).copy_(rnb_view.slice(1, 1-player, 2-player, 1));
        range_normalization_memory += value_normalization.view(range_normalization_memory.sizes());
    }

    aux_range_normalization.masked_fill_(aux_range_normalization.eq(0), 1);
    aux_next_round_serialized_range /= aux_range_normalization.view({-1, 1}).expand_as(aux_next_round_serialized_range);

    for (int player=0; player<constants.players_count; ++player) {
        torch::Tensor aux_next_round_inputs_slice = aux_next_round_inputs.slice(1, player * aux_bucket_count, (player + 1) * aux_bucket_count, 1);
        aux_next_round_inputs_slice.copy_(aux_next_round_extended_range.slice(1, player, player + 1, 1).slice(
                2, 0, aux_bucket_count, 1).view(aux_next_round_inputs_slice.sizes()));
    }

    torch::Tensor aux_serialized_inputs_view = aux_next_round_inputs.view({batch_size, -1});
    torch::Tensor aux_serialized_values_view = aux_next_round_values.view({batch_size, -1});

    aux_value_nn->get_value(aux_serialized_inputs_view, aux_serialized_values_view);

    if (use_memory) {
        range_normalization.masked_fill_(range_normalization.eq(0), 1);
        next_round_serialized_range /= range_normalization.view({range_normalization.sizes()[0], 1}).expand_as(next_round_serialized_range);
        for (int player=0; player<constants.players_count; ++player)
            next_round_inputs.slice(1, player*bucket_count, (player+1)*bucket_count, 1) = next_round_extended_range.slice(
                    1, player, player+1, 1).slice(2, 0, bucket_count, 1).squeeze();

        torch::Tensor serialized_inputs_view = next_round_inputs.view({batch_size, -1});
        torch::Tensor serialized_values_view = next_round_values.view({batch_size, -1});

        value_nn->get_value(serialized_inputs_view, serialized_values_view);
    }

    torch::Tensor aux_normalization_view = aux_value_normalization.view({batch_size, constants.players_count, 1});
    aux_next_round_values *= aux_normalization_view.expand_as(aux_next_round_values);

    if (use_memory) {
        torch::Tensor normalization_view = value_normalization.view({batch_size, constants.players_count, 1});
        next_round_values *= normalization_view.expand_as(next_round_values);
        counterfactual_value_memory += next_round_values;
    }

    torch::Tensor _aux_next_round_values = aux_next_round_values.view({batch_size * constants.players_count, -1});
    torch::Tensor _values = values.view({batch_size * constants.players_count, -1});
    _bucket_value_to_hand_value_aux(_aux_next_round_values, _values);
}

void NextRoundValue::get_value(torch::Tensor& ranges, torch::Tensor& values) {

    assert (ranges.sizes()[0] == batch_size);
    iter ++;
    if (iter == 0) {
//        initializing data structures
        next_round_inputs = torch::zeros({batch_size, board_count, (bucket_count * constants.players_count + 1)}, torch::kFloat32).to(device);
        next_round_values = torch::zeros({batch_size, board_count, constants.players_count, bucket_count}, torch::kFloat32).to(device);
        transposed_next_round_values = torch::zeros({batch_size, constants.players_count, board_count, bucket_count}, torch::kFloat32).to(device);
        next_round_extended_range = torch::zeros({batch_size, constants.players_count, board_count * bucket_count}, torch::kFloat32).to(device);
        next_round_serialized_range = next_round_extended_range.view({-1, bucket_count});
        value_normalization = torch::zeros({batch_size, constants.players_count, board_count}, torch::kFloat32).to(device);
//        handling pot feature for the nn
        next_round_inputs.slice(2, bucket_count * constants.players_count, max_size, 1).squeeze(2).copy_(
                pot_sizes.view({-1, 1}).expand({batch_size, board_count}) / params::stack);
    }

    bool use_memory = (iter >= cfr_skip_iters[street]);

    if (use_memory && (iter == cfr_skip_iters[street])) {
        range_normalization_memory = torch::zeros({batch_size * board_count * constants.players_count, 1}, torch::kFloat32).to(device);
        counterfactual_value_memory = torch::zeros({batch_size, constants.players_count, board_count, bucket_count}, torch::kFloat32).to(device);
    }

    torch::Tensor _ranges = ranges.view({batch_size * constants.players_count, -1});
    torch::Tensor _next_round_extended_range = next_round_extended_range.view({batch_size * constants.players_count, -1});
    _hand_range_to_bucket_range(_ranges, _next_round_extended_range);
    range_normalization = next_round_serialized_range.sum(1);
    torch::Tensor rn_view = range_normalization.view({batch_size, constants.players_count, board_count});

    for (int player=0; player<constants.players_count; ++player)
        value_normalization.slice(1, player, player+1, 1).copy_(rn_view.slice(1, 1-player, 2-player, 1));
    if (use_memory)
        range_normalization_memory += value_normalization.view(range_normalization_memory.sizes());
    range_normalization.masked_fill_(range_normalization.eq(0), 1);
    next_round_serialized_range /= range_normalization.view({-1, 1}).expand_as(next_round_serialized_range);

    for (int player=0; player<constants.players_count; ++player) {
        torch::Tensor next_round_inputs_slice = next_round_inputs.slice(2, player*bucket_count, (player+1)*bucket_count, 1);
        next_round_inputs_slice.copy_(next_round_extended_range.slice(1, player, player + 1, 1).view(next_round_inputs_slice.sizes()));
    }

    torch::Tensor serialized_inputs_view = next_round_inputs.view({batch_size * board_count, -1});
    torch::Tensor serialized_values_view = next_round_values.view({batch_size * board_count, -1});

    value_nn->get_value(serialized_inputs_view, serialized_values_view);

    torch::Tensor normalization_view = value_normalization.view({batch_size, constants.players_count, board_count, 1}).transpose(1, 2);
    next_round_values *= normalization_view.expand_as(next_round_values);

    transposed_next_round_values.copy_(next_round_values.transpose(1, 2));

    if (use_memory)
        counterfactual_value_memory += transposed_next_round_values;

    torch::Tensor _transposed_next_round_values = transposed_next_round_values.view({batch_size * constants.players_count, -1});
    torch::Tensor _values = values.view({batch_size * constants.players_count, -1});
    _bucket_value_to_hand_value(_transposed_next_round_values, _values);
}

void NextRoundValue::get_value_last_20(torch::Tensor &ranges, torch::Tensor &values) {
    assert (ranges.sizes()[0] == batch_size);

    iter ++;
    if (iter == 0 || iter == cfr_skip_iters[street]) {
//        initializing data structures
        next_round_inputs = torch::zeros({batch_size, board_count, (bucket_count * constants.players_count + 1)}, torch::kFloat32).to(device);
        next_round_values = torch::zeros({batch_size, board_count, constants.players_count, bucket_count}, torch::kFloat32).to(device);
        transposed_next_round_values = torch::zeros({batch_size, constants.players_count, board_count, bucket_count}, torch::kFloat32).to(device);
        next_round_extended_range = torch::zeros({batch_size, constants.players_count, board_count, bucket_count + 1}, torch::kFloat32).to(device);
        next_round_serialized_range = next_round_extended_range.view({-1, bucket_count + 1});
        values_per_board = torch::zeros({batch_size * constants.players_count, board_count, hand_count}, torch::kFloat32).to(device);
        value_normalization = torch::zeros({batch_size, constants.players_count, board_count}, torch::kFloat32).to(device);
//        handling pot feature for the nn
        next_round_inputs.slice(2, bucket_count * constants.players_count, max_size, 1).squeeze(2).copy_(
                pot_sizes.view({-1, 1}).expand({batch_size, board_count}) / params::stack);
    }

    bool use_memory = (iter >= cfr_skip_iters[street]);

    if (use_memory && (iter == cfr_skip_iters[street])) {
        range_normalization_memory = torch::zeros({batch_size * board_count * constants.players_count, 1}, torch::kFloat32).to(device);
        counterfactual_value_memory = torch::zeros({batch_size, constants.players_count, board_count, bucket_count}, torch::kFloat32).to(device);
    }

    torch::Tensor _ranges = ranges.view({batch_size * constants.players_count, -1});
    torch::Tensor _next_round_extended_range = next_round_extended_range.view({batch_size * constants.players_count, -1});
    _hand_range_to_bucket_range_last_20(_ranges, _next_round_extended_range);
    range_normalization = next_round_serialized_range.slice(1, 0, bucket_count + 1, 1).sum(1);
    torch::Tensor rn_view = range_normalization.view({batch_size, constants.players_count, board_count});

    for (int player=0; player<constants.players_count; ++player)
        value_normalization.slice(1, player, player+1, 1).copy_(rn_view.slice(1, 1-player, 2-player, 1));
    if (use_memory)
        range_normalization_memory += value_normalization.view(range_normalization_memory.sizes());
    range_normalization.masked_fill_(range_normalization.eq(0), 1);
    next_round_serialized_range /= range_normalization.view({-1, 1}).expand_as(next_round_serialized_range);

    for (int player=0; player<constants.players_count; ++player) {
        torch::Tensor next_round_inputs_slice = next_round_inputs.slice(2, player*bucket_count, (player+1)*bucket_count, 1);
        next_round_inputs_slice.copy_(next_round_extended_range.slice(1, player, player + 1, 1).slice(3, 0, bucket_count, 1).view(next_round_inputs_slice.sizes()));
    }

    torch::Tensor serialized_inputs_view = next_round_inputs.view({batch_size * board_count, -1});
    torch::Tensor serialized_values_view = next_round_values.view({batch_size * board_count, -1});

    value_nn->get_value(serialized_inputs_view, serialized_values_view);

    torch::Tensor normalization_view = value_normalization.view({batch_size, constants.players_count, board_count, 1}).transpose(1, 2);
    next_round_values *= normalization_view.expand_as(next_round_values);

    transposed_next_round_values.copy_(next_round_values.transpose(1, 2));

    if (use_memory)
        counterfactual_value_memory += transposed_next_round_values;

    torch::Tensor _transposed_next_round_values = transposed_next_round_values.view({batch_size * constants.players_count, -1});
    torch::Tensor _values = values.view({batch_size * constants.players_count, -1});
    _bucket_value_to_hand_value_last_20(_transposed_next_round_values, _values);
}

void NextRoundValue::get_value_on_board(Board& board, torch::Tensor& values) {
    assert (iter == cfr_iters[street] - 1);
    assert (values.sizes()[0] == batch_size);

    _prepare_next_round_values();

    if (street == 1) {
        torch::Tensor _counterfactual_value_memory = counterfactual_value_memory.view({batch_size * constants.players_count, -1});
        torch::Tensor _values = values.view({batch_size * constants.players_count, -1});
        _bucket_value_to_hand_value_on_board(board, _counterfactual_value_memory, _values);
    }
    else
        _bucket_value_to_hand_value_on_board(board, counterfactual_value_memory, values);
}

void NextRoundValue::_prepare_next_round_values() {
    assert (iter == cfr_iters[street] - 1);

    if (_values_are_prepared)
        return;

    range_normalization_memory.masked_fill_(range_normalization_memory.eq(0), 1);
    torch::Tensor serialized_memory_view = counterfactual_value_memory.view({-1, bucket_count});
    serialized_memory_view /= range_normalization_memory.expand_as(serialized_memory_view);

    _values_are_prepared = true;
}

NextRoundValue& get_flop_value() {
    static NextRoundValue flop_value(1, &get_flop_nn(), &get_aux_nn());
    return flop_value;
}

NextRoundValue& get_turn_value() {
    static NextRoundValue turn_value(2, &get_turn_nn());
    return turn_value;
}
