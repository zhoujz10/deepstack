//
// Created by zhou on 19-6-3.
//


#include "lookahead.h"


Lookahead::Lookahead(bool _is_next) {
    is_next = _is_next;
    builder = new LookaheadBuilder(this);
}

void Lookahead::build_lookahead(Node& _tree) {
    if (is_next)
        builder->build_from_tree(_tree, terminal_equity.river_hand_abstract_count);
    else {
        terminal_equity.set_board(tree->board, tree->limit_to_street);
        builder->build_from_tree(_tree);
    }
}

void Lookahead::resolve_first_node(torch::Tensor player_range, torch::Tensor opponent_range) {
    ranges_data[0].slice(3, 0, 1, 1).copy_(player_range);
    ranges_data[0].slice(3, 1, -1, 1).copy_(opponent_range);
    _compute();
}

void Lookahead::resolve(torch::Tensor player_range, torch::Tensor opponent_cfvs, torch::Tensor opponent_range_warm_start) {
//    reconstruction_gadget = CFRDGadget(self.tree['board'], player_range, opponent_cfvs, opponent_range_warm_start)
    ranges_data[0].slice(3, 0, 1, 1).copy_(player_range);
//    reconstruction_opponent_cfvs = opponent_cfvs
    _compute();
}

void Lookahead::_compute() {
//    for (int _iter=0; _iter<cfr_iters[tree->street]; ++_iter) {
//        _set_opponent_starting_range(_iter);
//        _compute_current_strategies();
//        _compute_ranges();
//        _compute_update_average_strategies(_iter);
//        _compute_terminal_equities(_iter);
//        _compute_cfvs();
//        _compute_regrets();
//        _compute_cumulate_average_cfvs(_iter);
//    }
//    _compute_normalize_average_strategies();
//    _compute_normalize_average_cfvs();
}

void Lookahead::_compute_current_strategies() {
    for (int d=1; d<depth+1; ++d) {
        positive_regrets_data[d].copy_(regrets_data[d].clamp(regret_epsilon, max_number));
        positive_regrets_data[d] *= empty_action_mask[d];

        regrets_sum[d].slice(2, 0, 1, 1).copy_(positive_regrets_data[d].sum(0));
        current_strategy_data[d].copy_(positive_regrets_data[d] / regrets_sum[d].slice(2, 0, 1, 1).expand_as(positive_regrets_data[d]));
    }

    if (tree->street == 3 && *std::max_element(tree->bets, tree->bets+1) < stack)
        river_lookahead->_compute_current_strategies_next_street();
}

void Lookahead::_compute_current_strategies_next_street() {
    for (int d=1; d<depth+1; ++d) {
        positive_regrets_data[d].copy_(regrets_data[d].clamp(regret_epsilon, max_number));

        positive_regrets_data[d] *= empty_action_mask[d];

        regrets_sum[d].slice(4, 0, 1, 1).copy_(positive_regrets_data[d].sum(0));
        current_strategy_data[d].copy_(positive_regrets_data[d] / regrets_sum[d].slice(4, 0, 1, 1).expand_as(positive_regrets_data[d]));
    }
}

void Lookahead::_compute_ranges() {
    for (int d=0; d<depth; ++d) {
        torch::Tensor& current_level_ranges = ranges_data[d];
        torch::Tensor& next_level_ranges = ranges_data[d+1];

        int prev_layer_terminal_actions_count = terminal_actions_count[d-1];
        int prev_layer_bets_count = bets_count[d-1];
        int gp_layer_nonallin_bets_count = nonallinbets_count[d-2];

        inner_nodes[d].copy_(current_level_ranges.slice(0, prev_layer_terminal_actions_count, -1, 1).slice(1, 0, gp_layer_nonallin_bets_count, 1));
        torch::Tensor super_view = inner_nodes[d].transpose(1,2).view({1, prev_layer_bets_count, -1, constants.players_count, hand_count});
        next_level_ranges.copy_(super_view.expand_as(next_level_ranges));

        next_level_ranges.slice(3, acting_player[d], acting_player[d]+1, 1) *= current_strategy_data[d+1];
    }

    if (tree->street == 3 && *std::max_element(tree->bets, tree->bets+1) < stack) {
        for (auto& t : next_street_lookahead) {
            auto layer = std::get<0>(t);
            auto action_id = std::get<1>(t);
            auto parent_id = std::get<2>(t);
            auto gp_id = std::get<3>(t);
            auto i = std::get<4>(t);

            torch::Tensor rd_slice = ranges_data[layer].slice(0, action_id, action_id+1, 1)
                    .slice(1, parent_id, parent_id+1, 1).slice(2, gp_id, gp_id+1, 1);
            river_lookahead->ranges_convert[i].copy_(torch::bmm(rd_slice.view({1,2,hand_count}).expand({48,-1,-1}), terminal_equity.river_hand_abstract));
            river_lookahead->ranges_data_hand[i].copy_(rd_slice.expand_as(river_lookahead->ranges_data_hand[i]));

            river_lookahead->ranges_data_hand[i].masked_fill_(terminal_equity.mask_next_street, 0);

            if (tree->current_player == 0)
                river_lookahead->ranges_data[0].slice(0, 0, 1, 1).slice(1, 0, 1, 1).slice(2, 0, 1, 1).copy_(river_lookahead->ranges_convert);
            else {
                river_lookahead->ranges_data[0].slice(0, 0, 1, 1).slice(1, 0, 1, 1).slice(2, 0, 1, 1).slice(5, 0, 1, 1).copy_(
                        river_lookahead->ranges_convert.slice(2, 1, -1, 1));
                river_lookahead->ranges_data[0].slice(0, 0, 1, 1).slice(1, 0, 1, 1).slice(2, 0, 1, 1).slice(5, 1, -1, 1).copy_(
                        river_lookahead->ranges_convert.slice(2, 0, 1, 1));
            }
            river_lookahead->_compute_ranges_next_street();
        }
    }
}

void Lookahead::_compute_ranges_next_street() {
    for (int d=0; d<depth; ++d) {
        torch::Tensor& current_level_ranges = ranges_data[d];
        torch::Tensor& next_level_ranges = ranges_data[d+1];

        int prev_layer_terminal_actions_count = terminal_actions_count[d-1];
        int prev_layer_bets_count = bets_count[d-1];
        int gp_layer_nonallin_bets_count = nonallinbets_count[d-2];

        inner_nodes[d].copy_(current_level_ranges.slice(0, prev_layer_terminal_actions_count, -1, 1).slice(1, 0, gp_layer_nonallin_bets_count, 1));
        torch::Tensor super_view;
        if (idx_range_by_depth.find(d) != idx_range_by_depth.end())
            super_view = inner_nodes[d].transpose(1,2).view({1, prev_layer_bets_count, -1, idx_range_by_depth[d],
                                                             boards_count[4], constants.players_count, terminal_equity.river_hand_abstract_count});
        else
            super_view = inner_nodes[d].transpose(1,2).view({1, prev_layer_bets_count, -1, river_count,
                                                             boards_count[4], constants.players_count, terminal_equity.river_hand_abstract_count});

        if (idx_range_by_depth.find(d+1) != idx_range_by_depth.end())
            next_level_ranges.copy_(super_view.slice(3, 0, idx_range_by_depth[d+1], 1).expand_as(next_level_ranges));
        else
            next_level_ranges.copy_(super_view.expand_as(next_level_ranges));

        next_level_ranges.slice(5, acting_player[d], acting_player[d]+1, 1) *= current_strategy_data[d+1];
    }
}




















