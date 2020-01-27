//
// Created by zhou on 19-6-12.
//


#include "cfrd_gadget.h"

CFRDGadget::CFRDGadget(Board& board, const torch::Tensor& opponent_cfvs, const torch::Tensor& src_opponent_range_warm_start) {

    input_opponent_value.copy_(opponent_cfvs);
    opponent_range_warm_start.copy_(src_opponent_range_warm_start);
    card_tools.get_possible_hand_indexes(board, range_mask);
}

void CFRDGadget::compute_opponent_range(const torch::Tensor& current_opponent_cfvs, int iteration) {

    if (warm_start && is_round_first_move && (iteration == 0)) {
        torch::Tensor uniform_range = torch::ones({hand_count}, torch::kFloat32).to(device) * range_mask;
        uniform_range /= uniform_range.sum();
        opponent_range_warm_start /= opponent_range_warm_start.sum();
        input_opponent_range.copy_(opponent_range_warm_start * 0.9 + uniform_range * 0.1);
        input_opponent_range *= range_mask;
        return;
    }


    torch::Tensor terminate_values = input_opponent_value;

    total_values.copy_(current_opponent_cfvs * play_current_strategy);
    total_values_p2.copy_(terminate_values * terminate_current_strategy);
    total_values += total_values_p2;

    play_current_regret.copy_(current_opponent_cfvs);
    play_current_regret -= total_values;

    terminate_current_regret.copy_(terminate_values);
    terminate_current_regret -= total_values;

    play_regrets += play_current_regret;
    terminate_regrets += terminate_current_regret;

    play_regrets.clamp_(gadget_epsilon, max_number);
    terminate_regrets.clamp_(gadget_epsilon, max_number);

    regret_sum.copy_(play_regrets);
    regret_sum += terminate_regrets;

    play_current_strategy.copy_(play_regrets);
    terminate_current_strategy.copy_(terminate_regrets);

    play_current_strategy /= regret_sum;
    terminate_current_strategy /= regret_sum;


//    if (warm_start && is_round_first_move && (iteration == 0)) {
//        torch::Tensor uniform_range = torch::ones({hand_count}, torch::kFloat32).to(device) * range_mask;
//        uniform_range /= uniform_range.sum();
//        opponent_range_warm_start /= opponent_range_warm_start.sum();
//        play_current_strategy.copy_(opponent_range_warm_start * 0.9 + uniform_range * 0.1);
//        terminate_current_strategy = -play_current_strategy + 1;
//
//
//        play_current_strategy *= range_mask;
//        terminate_current_strategy *= range_mask;
//        play_regrets.copy_(play_current_strategy);
//        terminate_regrets.copy_(terminate_current_strategy);
//        play_regrets.clamp_(gadget_epsilon, max_number);
//        terminate_regrets.clamp_(gadget_epsilon, max_number);
//    }
//    else {
//        play_current_strategy *= range_mask;
//        terminate_current_strategy *= range_mask;
//    }


    play_current_strategy *= range_mask;
    terminate_current_strategy *= range_mask;


    input_opponent_range.copy_(play_current_strategy);
//    if (warm_start && is_round_first_move && (iteration == 0)) {
//        torch::Tensor uniform_range = torch::ones({hand_count}, torch::kFloat32).to(device) * range_mask;
//        uniform_range /= uniform_range.sum();
//        input_opponent_range.copy_(opponent_range_warm_start * 0.9 + uniform_range * 0.1);
//    }
//    else
//        input_opponent_range.copy_(play_current_strategy);
}