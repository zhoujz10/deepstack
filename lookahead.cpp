//
// Created by zhou on 19-6-3.
//


#include "lookahead.h"















LookaheadBuilder::LookaheadBuilder(Lookahead& src) {
    lookahead = &src;
    lookahead->ccall_action_index = 0;
    lookahead->fold_action_index = 1;
}

void LookaheadBuilder::_construct_transition_boxes() {
    if (lookahead->tree->street >= 3)
        return;
    lookahead->num_pot_sizes = 0;
//    TODO: finish this function
}

void LookaheadBuilder::_compute_structure() {
    assert (1 <= lookahead->tree->street && lookahead->tree->street <= 4);

    lookahead->acting_player = torch::ones(lookahead->depth+1, torch::kInt32).to(device);
    lookahead->acting_player[0] = 0;
    for (int d=1; d<lookahead->depth+1; ++d)
        lookahead->acting_player[d] = 1 - lookahead->acting_player[d-1];
    lookahead->bets_count[-2] = 1;
    lookahead->bets_count[-1] = 1;
    lookahead->nonallinbets_count[-2] = 1;
    lookahead->nonallinbets_count[-1] = 1;
    lookahead->terminal_actions_count[-2] = 0;
    lookahead->terminal_actions_count[-1] = 0;
    lookahead->actions_count[-2] = 1;
    lookahead->actions_count[-1] = 1;

    lookahead->nonterminal_nodes_count[0] = 1;
    lookahead->nonterminal_nodes_count[1] = lookahead->bets_count[0];
    lookahead->nonterminal_nonallin_nodes_count[-1] = 1;
    lookahead->nonterminal_nonallin_nodes_count[0] = 1;
    lookahead->nonterminal_nonallin_nodes_count[1] = lookahead->nonterminal_nodes_count[1] - 1;
    lookahead->all_nodes_count[0] = 1;
    lookahead->all_nodes_count[1] = lookahead->actions_count[0];
    lookahead->terminal_nodes_count[0] = 0;
    lookahead->terminal_nodes_count[1] = 2;
    lookahead->allin_nodes_count[0] = 0;
    lookahead->allin_nodes_count[1] = 1;
    lookahead->inner_nodes_count[0] = 1;
    lookahead->inner_nodes_count[1] = 1;

    for (int d=1; d<lookahead->depth+1; ++d) {
        lookahead->all_nodes_count[d+1] = lookahead->nonterminal_nonallin_nodes_count[d-1] * lookahead->bets_count[d-1] * lookahead->actions_count[d];
        lookahead->allin_nodes_count[d+1] = lookahead->nonterminal_nonallin_nodes_count[d-1] * lookahead->bets_count[d-1] * 1;
        lookahead->nonterminal_nodes_count[d+1] = lookahead->nonterminal_nonallin_nodes_count[d-1] * lookahead->nonallinbets_count[d-1] * lookahead->bets_count[d];
        lookahead->nonterminal_nonallin_nodes_count[d+1] = lookahead->nonterminal_nonallin_nodes_count[d-1] * lookahead->nonallinbets_count[d-1] * lookahead->nonallinbets_count[d];
        lookahead->terminal_nodes_count[d+1] = lookahead->nonterminal_nonallin_nodes_count[d-1] * lookahead->bets_count[d-1] * lookahead->terminal_actions_count[d];
    }
}

void LookaheadBuilder::construct_data_structures() {
    _compute_structure();

    if (lookahead->is_next) {
        lookahead->ranges_data[0] = torch::ones({1, 1, 1, lookahead->river_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device) / river_hand_abstract_count;
        lookahead->ranges_data[1] = torch::ones({lookahead->actions_count[0], 1, 1, lookahead->river_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device) / river_hand_abstract_count;
        lookahead->pot_size[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32).to(device);
        lookahead->pot_size[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->cfvs_data[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32).to(device);
        lookahead->cfvs_data[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32).to(device);

        if (river_hand_abstract_count != hand_count) {
            lookahead->ranges_data_hand = torch::zeros({lookahead->river_count, boards_count[4], constants.players_count, hand_count}, torch::kFloat32).to(device);
            lookahead->ranges_convert = torch::zeros({lookahead->river_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);
            lookahead->cfvs_data_hand = torch::zeros({lookahead->river_count, boards_count[4], constants.players_count, hand_count}, torch::kFloat32).to(device);
            lookahead->cfvs_data_hand_memory = lookahead->cfvs_data_hand.clone();
            lookahead->ranges_data_hand_memory = lookahead->ranges_data_hand.clone();
        }

        lookahead->average_cfvs_data[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32);
        lookahead->average_cfvs_data[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32);
        lookahead->placeholder_data[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32);
        lookahead->placeholder_data[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32);

        lookahead->average_strategies_data[1] = torch::zeros({lookahead->actions_count[0], 1, 1, lookahead->river_count, boards_count[4], river_hand_abstract_count}, torch::kFloat32).to(device);
        lookahead->current_strategy_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->regrets_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->current_regrets_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->positive_regrets_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->empty_action_mask[1] = torch::ones(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);

        lookahead->regrets_sum[0] = torch::zeros({1, 1, lookahead->river_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);
        lookahead->regrets_sum[1] = torch::zeros({lookahead->bets_count[-1], 1, lookahead->river_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);

        lookahead->inner_nodes[0] = torch::zeros({1, 1, 1, lookahead->river_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);
        lookahead->swap_data[0] = lookahead->inner_nodes[0].transpose(1,2).clone();
        lookahead->inner_nodes_p1[0] = torch::zeros({1, 1, 1, lookahead->river_count, boards_count[4], 1, river_hand_abstract_count}, torch::kFloat32).to(device);

        if (lookahead->depth > 1) {
            lookahead->inner_nodes[1] = torch::zeros({lookahead->bets_count[0], 1, 1, lookahead->river_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);
            lookahead->swap_data[1] = lookahead->inner_nodes[1].transpose(1, 2).clone();
            lookahead->inner_nodes_p1[1] = torch::zeros({lookahead->bets_count[0], 1, 1, lookahead->river_count, boards_count[4], 1, river_hand_abstract_count}, torch::kFloat32).to(device);
        }

        for (int d=2; d<lookahead->depth+1; ++d) {
            int r_count = lookahead->river_count;
            auto it = lookahead->idx_range_by_depth.find(d);
            if (it != lookahead->idx_range_by_depth.end())
                r_count = it->second;

            lookahead->ranges_data[d] = torch::zeros({lookahead->actions_count[d-1], lookahead->bets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], r_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);
            lookahead->cfvs_data[d] = lookahead->ranges_data[d].clone();
            lookahead->placeholder_data[d] = lookahead->ranges_data[d].clone();
            lookahead->pot_size[d] = torch::ones(lookahead->ranges_data[d].sizes(), torch::kFloat32).to(device);

            lookahead->average_strategies_data[d] = torch::zeros({lookahead->actions_count[d-1], lookahead->bets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], r_count, boards_count[4], river_hand_abstract_count}, torch::kFloat32).to(device);
            lookahead->current_strategy_data[d] = lookahead->average_strategies_data[d].clone();
            lookahead->regrets_data[d] = torch::ones(lookahead->average_strategies_data[d].sizes(), torch::kFloat32).to(device) * regret_epsilon;
            lookahead->current_regrets_data[d] = torch::zeros(lookahead->average_strategies_data[d].sizes(), torch::kFloat32).to(device);
            lookahead->empty_action_mask[d] = torch::ones(lookahead->average_strategies_data[d].sizes(), torch::kFloat32).to(device);
            lookahead->positive_regrets_data[d] = lookahead->regrets_data[d].clone();

            lookahead->regrets_sum[d] = torch::zeros({lookahead->bets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], r_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);

            if (d < lookahead->depth) {
                lookahead->inner_nodes[d] = torch::zeros({lookahead->bets_count[d-1], lookahead->nonallinbets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], r_count, boards_count[4], constants.players_count, river_hand_abstract_count}, torch::kFloat32).to(device);
                lookahead->inner_nodes_p1[d] = torch::zeros({lookahead->bets_count[d-1], lookahead->nonallinbets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], r_count, boards_count[4], 1, river_hand_abstract_count}, torch::kFloat32).to(device);
                lookahead->swap_data[d] = lookahead->inner_nodes[d].transpose(1,2).clone();
            }
        }
    }
    else {

    }
}
































