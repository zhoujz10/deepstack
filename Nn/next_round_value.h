//
// Created by zhou on 19-6-10.
//

#ifndef DEEPSTACK_CPP_NEXT_ROUND_VALUE_H
#define DEEPSTACK_CPP_NEXT_ROUND_VALUE_H


#include <torch/torch.h>
#include "../Game/board.h"
#include "../Game/card_tools.h"
#include "value_nn.h"


class NextRoundValue {

public:
    CardTools& card_tools = get_card_tools();
    ValueNn* value_nn = nullptr;
    ValueNn* aux_value_nn = nullptr;
    int street = 1;
    int iter = -1;
    int batch_size = 0;
    int aux_bucket_count = 169;
    int bucket_count = 1000;
    int board_count = 0;
    float weight_constant = 0;

    uint32_t *canonical_map_turn = nullptr;
    uint16_t *assignments_turn = nullptr;

    torch::Tensor board_buckets;
    torch::Tensor impossible_mask;
    torch::Tensor board_indexes;
    torch::Tensor board_indexes_scatter;
    torch::Tensor values_per_board;
    torch::Tensor aux_values_per_board;
    torch::Tensor aux_next_round_inputs;
    torch::Tensor aux_next_round_values;
    torch::Tensor aux_next_round_extended_range;
    torch::Tensor aux_next_round_serialized_range;
    torch::Tensor aux_value_normalization;
    torch::Tensor bucket_range;
    torch::Tensor range_normalization_memory;
    torch::Tensor counterfactual_value_memory;
    torch::Tensor aux_range_normalization;

    torch::Tensor transposed_next_round_values;

    torch::Tensor pot_sizes;
    torch::Tensor next_round_inputs;
    torch::Tensor next_round_values;
    torch::Tensor next_round_extended_range;
    torch::Tensor next_round_serialized_range;
    torch::Tensor range_normalization;
    torch::Tensor value_normalization;


    NextRoundValue(int _street, ValueNn *src_value_nn, ValueNn *src_aux_nn = nullptr);
    void init_bucketing(Board* board_ptr = nullptr);
    void start_computation(torch::Tensor& src_pot_sizes);
    void get_value_aux(torch::Tensor& ranges, torch::Tensor& values, int next_board_idx=-1);
    void get_value(torch::Tensor& ranges, torch::Tensor& values);
    void get_value_on_board(Board& board, torch::Tensor& values);

    void init_var();

private:

    bool _values_are_prepared = false;

    torch::Tensor _range_matrix;
    torch::Tensor _range_matrix_board_view;
    torch::Tensor _aux_range_matrix;
    torch::Tensor _aux_reverse_value_matrix;
    torch::Tensor _reverse_value_matrix;

    void _hand_range_to_bucket_range(torch::Tensor& hand_range, torch::Tensor& bucket_range);
    void _bucket_value_to_hand_value(torch::Tensor& bucket_value, torch::Tensor& hand_value);
    void _hand_range_to_bucket_range_on_board(int board_idx, torch::Tensor& hand_range, torch::Tensor& bucket_range);
    void _bucket_value_to_hand_value_on_board(Board& board, torch::Tensor& bucket_value, torch::Tensor& hand_value);
    void _hand_range_to_bucket_range_aux(const torch::Tensor& hand_range, torch::Tensor& bucket_range);
    void _bucket_value_to_hand_value_aux(const torch::Tensor& bucket_value, torch::Tensor& hand_value);

    void _prepare_next_round_values();
};

NextRoundValue& get_flop_value();
NextRoundValue& get_turn_value();

#endif //DEEPSTACK_CPP_NEXT_ROUND_VALUE_H
