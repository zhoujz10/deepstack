//
// Created by zhou on 19-6-6.
//


#include <cmath>
#include "lookahead.h"


LookaheadBuilder::LookaheadBuilder(Lookahead *ptr) {
    lookahead = ptr;
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

    lookahead->acting_player_tensor = torch::ones(lookahead->depth+1, torch::kInt32).to(device);
    lookahead->acting_player_tensor[0] = 0;
    lookahead->acting_player.push_back(0);
    for (int d=1; d<lookahead->depth+1; ++d) {
        lookahead->acting_player_tensor[d] = 1 - lookahead->acting_player_tensor[d - 1];
        lookahead->acting_player.push_back(1 - lookahead->acting_player[-1]);
    }
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
        lookahead->ranges_data[0] = torch::ones({1, 1, 1, constants.players_count, hand_count}, torch::kFloat32).to(device) / hand_count;
        lookahead->ranges_data[1] = torch::ones({lookahead->actions_count[0], 1, 1, constants.players_count, hand_count}, torch::kFloat32).to(device) / hand_count;
        lookahead->pot_size[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32).to(device);
        lookahead->pot_size[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->cfvs_data[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32).to(device);
        lookahead->cfvs_data[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->average_cfvs_data[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32).to(device);
        lookahead->average_cfvs_data[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->placeholder_data[0] = torch::zeros(lookahead->ranges_data[0].sizes(), torch::kFloat32).to(device);
        lookahead->placeholder_data[1] = torch::zeros(lookahead->ranges_data[1].sizes(), torch::kFloat32).to(device);

        lookahead->average_strategies_data[1] = torch::zeros({lookahead->actions_count[0], 1, 1, hand_count}, torch::kFloat32).to(device);
        lookahead->current_strategy_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->regrets_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->current_regrets_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->positive_regrets_data[1] = torch::zeros(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);
        lookahead->empty_action_mask[1] = torch::ones(lookahead->average_strategies_data[1].sizes(), torch::kFloat32).to(device);

        lookahead->regrets_sum[0] = torch::zeros({1, 1, constants.players_count, hand_count}, torch::kFloat32).to(device);
        lookahead->regrets_sum[1] = torch::zeros({lookahead->bets_count[-1], 1, constants.players_count, hand_count}, torch::kFloat32).to(device);

        lookahead->inner_nodes[0] = torch::zeros({1, 1, 1, constants.players_count, hand_count}, torch::kFloat32).to(device);
        lookahead->swap_data[0] = lookahead->inner_nodes[0].transpose(1,2).clone();
        lookahead->inner_nodes_p1[0] = torch::zeros({1, 1, 1, 1, hand_count}, torch::kFloat32).to(device);

        if (lookahead->depth > 1) {
            lookahead->inner_nodes[1] = torch::zeros({lookahead->bets_count[0], 1, 1, constants.players_count, hand_count}, torch::kFloat32).to(device);
            lookahead->swap_data[1] = lookahead->inner_nodes[1].transpose(1,2).clone();
            lookahead->inner_nodes_p1[1] = torch::zeros({lookahead->bets_count[0], 1, 1, 1, hand_count}, torch::kFloat32).to(device);
        }

        for (int d=2; d<lookahead->depth+1; ++d) {
            lookahead->ranges_data[d] = torch::zeros({lookahead->actions_count[d-1], lookahead->bets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], constants.players_count, hand_count}, torch::kFloat32).to(device);
            lookahead->cfvs_data[d] = lookahead->ranges_data[d].clone();
            lookahead->placeholder_data[d] = lookahead->ranges_data[d].clone();
            lookahead->pot_size[d] = torch::ones(lookahead->ranges_data[d].sizes(), torch::kFloat32).to(device) * stack;

            lookahead->average_strategies_data[d] = torch::zeros({lookahead->actions_count[d-1], lookahead->bets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], hand_count}, torch::kFloat32).to(device);
            lookahead->current_strategy_data[d] = lookahead->average_strategies_data[d].clone();
            lookahead->regrets_data[d] = torch::ones(lookahead->average_strategies_data[d].sizes(), torch::kFloat32).to(device) * regret_epsilon;
            lookahead->current_regrets_data[d] = torch::zeros(lookahead->average_strategies_data[d].sizes(), torch::kFloat32).to(device);
            lookahead->empty_action_mask[d] = torch::ones(lookahead->average_strategies_data[d].sizes(), torch::kFloat32).to(device);
            lookahead->positive_regrets_data[d] = lookahead->regrets_data[d].clone();

            lookahead->regrets_sum[d] = torch::zeros({lookahead->bets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], constants.players_count, hand_count}, torch::kFloat32).to(device);

            if (d < lookahead->depth) {
                lookahead->inner_nodes[d] = torch::zeros({lookahead->bets_count[d-1], lookahead->nonallinbets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], constants.players_count, hand_count}, torch::kFloat32).to(device);
                lookahead->inner_nodes_p1[d] = torch::zeros({lookahead->bets_count[d-1], lookahead->nonallinbets_count[d-2], lookahead->nonterminal_nonallin_nodes_count[d-2], 1, hand_count}, torch::kFloat32).to(device);
                lookahead->swap_data[d] = lookahead->inner_nodes[d].transpose(1,2).clone();
            }
        }
    }

    if (lookahead->is_next) {
        for (int d=1; d<lookahead->depth+1; ++d)
            if (d > 1 || lookahead->first_call_terminal) {
                int before = lookahead->num_term_call_nodes;
                lookahead->num_term_call_nodes += lookahead->ranges_data[d][1].sizes()[0] * lookahead->ranges_data[d][1].sizes()[1] * lookahead->ranges_data[d][1].sizes()[2] * boards_count[4];
                lookahead->term_call_indices[d] = std::pair<int, int>(before, lookahead->num_term_call_nodes);
            }

        for (int d=1; d<lookahead->depth+1; ++d) {
            int before = lookahead->num_term_fold_nodes;
            lookahead->num_term_fold_nodes += lookahead->ranges_data[d][0].sizes()[0] * lookahead->ranges_data[d][0].sizes()[1] * lookahead->ranges_data[d][0].sizes()[2] * boards_count[4];
            lookahead->term_fold_indices[d] = std::pair<int, int>(before, lookahead->num_term_fold_nodes);
        }

        lookahead->ranges_data_call = torch::zeros({lookahead->num_term_call_nodes, 2, river_hand_abstract_count}, torch::kFloat32).to(device);
        lookahead->ranges_data_fold = torch::zeros({lookahead->num_term_fold_nodes, 2, river_hand_abstract_count}, torch::kFloat32).to(device);

        lookahead->cfvs_data_call = torch::zeros({lookahead->num_term_call_nodes, 2, river_hand_abstract_count}, torch::kFloat32).to(device);
        lookahead->cfvs_data_fold = torch::zeros({lookahead->num_term_fold_nodes, 2, river_hand_abstract_count}, torch::kFloat32).to(device);
    }
    else {
        for (int d=1; d<lookahead->depth+1; ++d)
            if (d > 1 || lookahead->first_call_terminal) {
                int before = lookahead->num_term_call_nodes;
                if (lookahead->tree->street != 4) {
                    lookahead->num_term_call_nodes += lookahead->ranges_data[d][1][-1].sizes()[0];
                    lookahead->term_call_indices[d] = std::pair<int, int>(before, lookahead->num_term_call_nodes);
                }
                else {
                    lookahead->num_term_call_nodes += lookahead->ranges_data[d][1].sizes()[0] * lookahead->ranges_data[d][1].sizes()[1];
                    lookahead->term_call_indices[d] = std::pair<int, int>(before, lookahead->num_term_call_nodes);
                }
            }

        for (int d=1; d<lookahead->depth+1; ++d) {
            int before = lookahead->num_term_fold_nodes;
            lookahead->num_term_fold_nodes += lookahead->ranges_data[d][0].sizes()[0] * lookahead->ranges_data[d][0].sizes()[1];
            lookahead->term_fold_indices[d] = std::pair<int, int>(before, lookahead->num_term_fold_nodes);
        }

        lookahead->ranges_data_call = torch::zeros({lookahead->num_term_call_nodes, 2, hand_count}, torch::kFloat32).to(device);
        lookahead->ranges_data_fold = torch::zeros({lookahead->num_term_fold_nodes, 2, hand_count}, torch::kFloat32).to(device);

        lookahead->cfvs_data_call = torch::zeros({lookahead->num_term_call_nodes, 2, hand_count}, torch::kFloat32).to(device);
        lookahead->cfvs_data_fold = torch::zeros({lookahead->num_term_fold_nodes, 2, hand_count}, torch::kFloat32).to(device);
    }
}

void LookaheadBuilder::set_datastructures_from_tree_dfs(Node& node, const int layer, const int action_id, const int parent_id,
                                                        const int gp_id, const int cur_action_id, const int parent_action_id) {
    assert (node.pot > 0);
    if (lookahead->is_next) {
        for (int j=0; j<lookahead->river_count; ++j) {
            if (lookahead->idx_range_by_depth.find(layer) != lookahead->idx_range_by_depth.end() && j >= lookahead->idx_range_by_depth[layer])
                continue;
            lookahead->pot_size[layer][action_id][parent_id][gp_id][j] = *std::min_element(node.all_river_bets[j], node.all_river_bets[j]+1);
        }
    }
    else
        lookahead->pot_size[layer][action_id][parent_id][gp_id] = node.pot;

    if (layer == 2 && cur_action_id == constants.actions.ccall)
        lookahead->parent_action_id[parent_id] = parent_action_id;

    if (node.current_player == constants.players.chance)
        assert (parent_id <= lookahead->nonallinbets_count[layer - 2] - 1);

    if (layer < lookahead->depth + 1) {
        int gp_nonallinbets_count = lookahead->nonallinbets_count[layer-2];
        int prev_layer_terminal_actions_count = lookahead->terminal_actions_count[layer-1];

        int next_parent_id = action_id - prev_layer_terminal_actions_count;
        int next_gp_id = gp_id * gp_nonallinbets_count + parent_id;

        if (!node.terminal && (node.current_player == constants.players.chance)) {

            assert (parent_id <= lookahead->nonallinbets_count[layer - 2] - 1);
            bool node_with_empty_actions = node.children->size() < lookahead->actions_count[layer];

            if (node_with_empty_actions) {

                assert (layer > 0);
                int terminal_actions_count = lookahead->terminal_actions_count[layer];
                assert (terminal_actions_count == 2);
                int existing_bets_count = node.children->size() - terminal_actions_count;

                if (existing_bets_count == 0)
                    assert (action_id == lookahead->actions_count[layer - 1] - 1);

                for (int child_id = 0; child_id < terminal_actions_count; ++child_id) {
                    auto child_node = (*node.children)[child_id];
                    set_datastructures_from_tree_dfs(child_node, layer + 1, child_id, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id);
                }

                for (int b = 0; b < existing_bets_count; ++b)
                    set_datastructures_from_tree_dfs((*node.children)[node.children->size() - b - 1], layer + 1, lookahead->actions_count[layer] - b - 1,
                                                     next_parent_id, next_gp_id, node.actions[node.children->size() - b - 1], cur_action_id);

                if (existing_bets_count == 0)
                    lookahead->empty_action_mask[layer + 1].slice(0, terminal_actions_count, -1, 1)[next_parent_id][next_gp_id] = 0;
                else {
                    lookahead->empty_action_mask[layer + 1].slice(0, terminal_actions_count, -existing_bets_count, 1)[next_parent_id][next_gp_id] = 0;

                    if (lookahead->is_next)
                        for (int j = 1; j < lookahead->river_count; ++j) {
                            if (lookahead->idx_range_by_depth.find(layer + 1) != lookahead->idx_range_by_depth.end() &&
                                j >= lookahead->idx_range_by_depth[layer + 1])
                                continue;
                            for (int b = 0; b < existing_bets_count; ++b) {
                                auto child_node = (*node.children)[(*node.children).size() - b - 1];
                                if (!child_node.all_river_valid[j])
                                    lookahead->empty_action_mask[layer + 1][lookahead->actions_count[layer] - b - 1][next_parent_id][next_gp_id][j] = 0;
                            }
                        }
                }
            }
            else {
                for (int child_id = 0; child_id < node.children->size(); ++child_id) {
                    auto child_node = (*node.children)[child_id];
                    set_datastructures_from_tree_dfs(child_node, layer + 1, child_id, next_parent_id, next_gp_id, node.actions[child_id], cur_action_id);
                }

                if (lookahead->is_next) {
                    int terminal_actions_count = lookahead->terminal_actions_count[layer];
                    int existing_bets_count = node.children->size() - terminal_actions_count;
                    for (int j = 1; j < lookahead->river_count; ++j) {
                        if (lookahead->idx_range_by_depth.find(layer + 1) != lookahead->idx_range_by_depth.end() &&
                            j >= lookahead->idx_range_by_depth[layer + 1])
                            continue;
                        for (int b = 0; b < existing_bets_count; ++b) {
                            auto child_node = (*node.children)[(*node.children).size() - b - 1];
                            if (!child_node.all_river_valid[j])
                                lookahead->empty_action_mask[layer + 1][lookahead->actions_count[layer] - b - 1][next_parent_id][next_gp_id][j] = 0;
                        }
                    }
                }
            }
        }
        else if (node.street == 3 && (node.current_player == constants.players.chance) && node.next_street_root != nullptr) {
            lookahead->next_street_lookahead.push_back(
                    std::tuple<int, int, int, int, int, int>
                            (layer, action_id, parent_id, gp_id, node.river_idx, lookahead->tree->river_pots[node.river_idx]));
        }
    }
}

void LookaheadBuilder::build_from_tree(Node& tree, const int _river_hand_abstract_count) {
    river_hand_abstract_count = _river_hand_abstract_count;

    lookahead->tree = &tree;
    lookahead->depth = tree.depth;

    std::vector<Node*> t = { &tree };
    _compute_tree_structures(t, 0);

    lookahead->first_call_terminal = (*tree.children)[1].terminal;
    lookahead->first_call_transition = (*tree.children)[1].current_player == constants.players.chance;
    lookahead->first_call_check = (!lookahead->first_call_terminal) && (!lookahead->first_call_transition);

    construct_data_structures();

    set_datastructures_from_tree_dfs(tree, 0, 0, 0, 0, 0, -100);

    if (tree.street == 3 && *std::max_element(tree.bets, tree.bets+1) < stack) {
        lookahead->river_count = tree.river_count;
        lookahead->river_lookahead = new Lookahead(true);

        for (auto pot : lookahead->tree->river_pots) {
            if (pot == 0)
                break;
            lookahead->max_depth.push_back((int)ceil(log((double)stack / pot) / log(3)) + 2);
        }

        for (int d=4; d<*std::max_element(lookahead->max_depth.begin(), lookahead->max_depth.end())+1; ++d)
            for (auto it=lookahead->max_depth.begin(); it!=lookahead->max_depth.end(); ++it)
                if (*it < d) {
                    lookahead->river_lookahead->idx_range_by_depth[d] = it - lookahead->max_depth.begin();
                    break;
                }

        lookahead->river_lookahead->river_count = lookahead->river_count;
        lookahead->river_lookahead->build_lookahead(*tree.river_tree_node->next_street_root);
    }

    assert (lookahead->terminal_actions_count[0] == 1 || lookahead->terminal_actions_count[0] == 2);

    if (tree.bets[0] == tree.bets[1])
        lookahead->empty_action_mask[1][0] = 0;

    _construct_transition_boxes();
}



void LookaheadBuilder::_compute_tree_structures(std::vector<Node*>& current_layer, const int current_depth) {

    int layer_actions_count = 0;
    int layer_terminal_actions_count = 0;
    std::vector<Node*> next_layer;

    for (auto node_ptr : current_layer) {
        layer_actions_count = std::max(layer_actions_count, (int)node_ptr->children->size());

        int node_terminal_actions_count = 0;
        for (auto& child : *node_ptr->children)
            if (child.terminal || child.current_player == constants.players.chance)
                node_terminal_actions_count ++;

        layer_terminal_actions_count = std::max(layer_terminal_actions_count, node_terminal_actions_count);

        if (!node_ptr->terminal)
            for (auto& child : *node_ptr->children)
                next_layer.push_back(&child);
    }

    assert ((layer_actions_count == 0) == (!next_layer.empty()));
    assert ((layer_actions_count == 0) == (current_depth == lookahead->depth));

    lookahead->bets_count[current_depth] = layer_actions_count - layer_terminal_actions_count;

    lookahead->nonallinbets_count[current_depth] = layer_actions_count - layer_terminal_actions_count - 1;

    if (layer_actions_count == 2) {
        assert (layer_actions_count == layer_terminal_actions_count);
        lookahead->nonallinbets_count[current_depth] = 0;
    }

    lookahead->terminal_actions_count[current_depth] = layer_terminal_actions_count;
    lookahead->actions_count[current_depth] = layer_actions_count;

    if (!next_layer.empty()) {
        assert (layer_actions_count >= 2);
        _compute_tree_structures(next_layer, current_depth + 1);
    }
}

