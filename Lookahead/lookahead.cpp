//
// Created by zhou on 19-6-3.
//


#include "lookahead.h"


Lookahead::~Lookahead() {
    builder->lookahead = nullptr;
    delete river_lookahead;
//    std::cout << "Lookahead released." << std::endl;
    delete builder;
}

Lookahead::Lookahead(bool _is_next) {
    is_next = _is_next;
    builder = new LookaheadBuilder(this);
}

void Lookahead::build_lookahead(Node& _tree) {
    if (is_next)
        builder->build_from_tree(_tree, terminal_equity.river_hand_abstract_count);
    else {
        terminal_equity.set_board(_tree.board);
        builder->build_from_tree(_tree);
    }
}

void Lookahead::resolve_first_node(torch::Tensor& player_range, torch::Tensor& opponent_range) {
    ranges_data[0].slice(3, 0, 1, 1).copy_(player_range);
    ranges_data[0].slice(3, 1, 2, 1).copy_(opponent_range);
    _compute();
}

void Lookahead::resolve(torch::Tensor& player_range, torch::Tensor& opponent_cfvs, torch::Tensor& opponent_range_warm_start) {
    reconstruction_gadget = new CFRDGadget(tree->board, opponent_cfvs, opponent_range_warm_start);
    ranges_data[0].slice(3, 0, 1, 1).copy_(player_range);
    reconstruction_opponent_cfvs = true;
    _compute();
}

void Lookahead::_compute() {
//    std::cout << "start computing..." << std::endl;
    for (int _iter=0; _iter<cfr_iters[tree->street]; ++_iter) {
        _set_opponent_starting_range(_iter);
        _compute_current_strategies();
        _compute_ranges();
        _compute_update_average_strategies(_iter);
        _compute_terminal_equities(_iter);
        _compute_cfvs();
        _compute_regrets();
        _compute_cumulate_average_cfvs(_iter);
    }
    _compute_normalize_average_strategies();
    _compute_normalize_average_cfvs();
}

void Lookahead::_compute_current_strategies() {
    for (int d=1; d<depth+1; ++d) {
        positive_regrets_data[d].copy_(regrets_data[d].clamp(regret_epsilon, max_number));
        positive_regrets_data[d] *= empty_action_mask[d];

        regrets_sum[d].slice(2, 0, 1, 1).copy_(positive_regrets_data[d].sum(0).unsqueeze(2));
        current_strategy_data[d].copy_(positive_regrets_data[d] / regrets_sum[d].slice(2, 0, 1, 1).squeeze(2).expand_as(positive_regrets_data[d]));
    }

    if (tree->street == 3 && *std::max_element(tree->bets, tree->bets+2) < params::stack)
        river_lookahead->_compute_current_strategies_next_street();
}

void Lookahead::_compute_current_strategies_next_street() {
    for (int d=1; d<depth+1; ++d) {
        positive_regrets_data[d].copy_(regrets_data[d].clamp(regret_epsilon, max_number));

        positive_regrets_data[d] *= empty_action_mask[d];

        regrets_sum[d].slice(4, 0, 1, 1).copy_(positive_regrets_data[d].sum(0).unsqueeze(4));
        current_strategy_data[d].copy_(positive_regrets_data[d] / regrets_sum[d].slice(4, 0, 1, 1).squeeze(4).expand_as(positive_regrets_data[d]));
    }
}

void Lookahead::_compute_ranges() {
    for (int d=0; d<depth; ++d) {
        torch::Tensor& current_level_ranges = ranges_data[d];
        torch::Tensor& next_level_ranges = ranges_data[d+1];

        int prev_layer_terminal_actions_count = terminal_actions_count[d-1];
        int prev_layer_bets_count = bets_count[d-1];
        int gp_layer_nonallin_bets_count = nonallinbets_count[d-2];

        inner_nodes[d].copy_(current_level_ranges.slice(
                0, prev_layer_terminal_actions_count, max_size, 1).slice(1, 0, gp_layer_nonallin_bets_count, 1).squeeze(0));
//        torch::Tensor super_view = inner_nodes[d].transpose(1,2).view({1, prev_layer_bets_count, -1, constants.players_count, hand_count});
        next_level_ranges.copy_(inner_nodes[d].transpose(1,2).contiguous().view({1, prev_layer_bets_count, -1, constants.players_count, hand_count}).expand_as(next_level_ranges));

        next_level_ranges.slice(3, acting_player[d], acting_player[d]+1, 1) *= current_strategy_data[d+1].unsqueeze(3);
    }

    if (tree->street == 3 && *std::max_element(tree->bets, tree->bets+2) < params::stack) {

        for (auto& t : next_street_lookahead) {
            auto layer = std::get<0>(t);
            auto action_id = std::get<1>(t);
            auto parent_id = std::get<2>(t);
            auto gp_id = std::get<3>(t);
            auto i = std::get<4>(t);

            torch::Tensor rd_slice = ranges_data[layer][action_id][parent_id][gp_id];

//            std::cout << layer << ' ' << action_id << ' ' << parent_id << ' ' << gp_id << ' ' << i << std::endl;
//            std::cout << river_lookahead->ranges_convert.sizes() << std::endl;

            river_lookahead->ranges_convert[i].copy_(
                    torch::bmm(rd_slice.expand({boards_count[4], -1, -1}), terminal_equity.river_hand_abstract));
            river_lookahead->ranges_data_hand[i].copy_(rd_slice.expand_as(river_lookahead->ranges_data_hand[i]));

            river_lookahead->ranges_data_hand[i].squeeze().masked_fill_(terminal_equity.mask_next_street, 0);
        }

        if (tree->current_player == 0)
            river_lookahead->ranges_data[0][0][0][0].copy_(river_lookahead->ranges_convert);
        else {
            river_lookahead->ranges_data[0][0][0][0].slice(2, 0, 1, 1).copy_(
                    river_lookahead->ranges_convert.slice(2, 1, max_size, 1));
            river_lookahead->ranges_data[0][0][0][0].slice(2, 1, max_size, 1).copy_(
                    river_lookahead->ranges_convert.slice(2, 0, 1, 1));
        }

        river_lookahead->_compute_ranges_next_street();
    }
}

void Lookahead::_compute_ranges_next_street() {
    for (int d=0; d<depth; ++d) {
        torch::Tensor& current_level_ranges = ranges_data[d];
        torch::Tensor& next_level_ranges = ranges_data[d+1];

        int prev_layer_terminal_actions_count = terminal_actions_count[d-1];
        int prev_layer_bets_count = bets_count[d-1];
        int gp_layer_nonallin_bets_count = nonallinbets_count[d-2];

        inner_nodes[d].copy_(current_level_ranges.slice(
                0, prev_layer_terminal_actions_count, max_size, 1).slice(1, 0, gp_layer_nonallin_bets_count, 1).squeeze(0));
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

        next_level_ranges.slice(5, acting_player[d], acting_player[d]+1, 1).squeeze(5) *= current_strategy_data[d+1];
    }
}

void Lookahead::_compute_update_average_strategies(const int _iter) {
    if (_iter >= cfr_skip_iters[tree->street])
        average_strategies_data[1] += current_strategy_data[1];
}

void Lookahead::_compute_terminal_equities_terminal_equity() {

//    for (int d=1; d<depth+1; ++d) {
//        if (d>1 or first_call_terminal) {
//            if (tree->street != 4)
//                ranges_data_call.slice(0, term_call_indices[d].first*2, term_call_indices[d].second*2, 1).copy_(
//                        ranges_data[d][1][-1].view({-1, hand_count}));
//            else
//                ranges_data_call.slice(0, term_call_indices[d].first*2, term_call_indices[d].second*2, 1).copy_(
//                        ranges_data[d][1].view({term_call_indices[d].second*2 - term_call_indices[d].first*2, hand_count}));
//        }
//        ranges_data_fold.slice(0, term_fold_indices[d].first*2, term_fold_indices[d].second*2, 1).copy_(
//                ranges_data[d][0].view({term_fold_indices[d].second*2 - term_fold_indices[d].first*2, hand_count}));
//    }
//
//    terminal_equity.call_value(ranges_data_call, cfvs_data_call);
//    terminal_equity.fold_value(ranges_data_fold, cfvs_data_fold);
//
//    for (int d=1; d<depth+1; ++d) {
//        if (d>1 or first_call_terminal) {
//            if (tree->street != 4)
//                cfvs_data[d][1].slice(0, cfvs_data[d].sizes()[1]-1, cfvs_data[d].sizes()[1], 1).copy_(
//                        cfvs_data_call.slice(0, term_call_indices[d].first*2, term_call_indices[d].second*2, 1).view(
//                                cfvs_data[d][1].slice(0, cfvs_data[d].sizes()[1]-1, cfvs_data[d].sizes()[1], 1).sizes()));
//            else
//                cfvs_data[d][1].copy_(cfvs_data_call.slice(
//                        0, term_call_indices[d].first*2, term_call_indices[d].second*2, 1).view(cfvs_data[d][1].sizes()));
//        }
//        cfvs_data[d][0].copy_(cfvs_data_fold.slice(
//                0, term_fold_indices[d].first*2, term_fold_indices[d].second*2, 1).view(cfvs_data[d][0].sizes()));
//
//        int fold_mutliplier = acting_player[d] * 2 - 1;
//        cfvs_data[d][0].slice(2, 0, 1, 1) *= fold_mutliplier;
//        cfvs_data[d][0].slice(2, 1, max_size, 1) *= -fold_mutliplier;
//    }

    for (int d=1; d<depth+1; ++d) {

        if (tree->street <= 3) {
            if (d > 1 || first_call_terminal) {
                torch::Tensor _ranges_data = ranges_data[d][1][-1].view({-1, hand_count});
                torch::Tensor _cfvs_data = cfvs_data[d][1][-1].view({-1, hand_count});
                terminal_equity.call_value(_ranges_data, _cfvs_data);
            }
        }
        else {
            assert (tree->street == 4);
            if (d > 1 || first_call_terminal) {
                torch::Tensor _ranges_data = ranges_data[d][1].view({-1, hand_count});
                torch::Tensor _cfvs_data = cfvs_data[d][1].view({-1, hand_count});
                terminal_equity.call_value(_ranges_data, _cfvs_data);
            }
        }

        int fold_mutliplier = acting_player[d] * 2 - 1;

        torch::Tensor _ranges_data = ranges_data[d][0].view({-1, hand_count});
        torch::Tensor _cfvs_data = cfvs_data[d][0].view({-1, hand_count});
        terminal_equity.fold_value(_ranges_data, _cfvs_data);
        cfvs_data[d][0].slice(2, 0, 1, 1) *= fold_mutliplier;
        cfvs_data[d][0].slice(2, 1, 2, 1) *= -fold_mutliplier;
    }
}

void Lookahead::_compute_terminal_equities_terminal_equity_next_street() {
    for (int d=1; d<depth+1; ++d) {
        if (d>1 or first_call_terminal) {
            torch::Tensor _ranges_data = ranges_data[d][1].view({-1, terminal_equity.river_hand_abstract_count});
            torch::Tensor _cfvs_data = cfvs_data[d][1].view({-1, terminal_equity.river_hand_abstract_count});
            terminal_equity.call_value_next_street(_ranges_data, _cfvs_data);

            int fold_mutliplier = acting_player[d]*2 - 1;
            _ranges_data = ranges_data[d][0].view({-1, terminal_equity.river_hand_abstract_count});
            _cfvs_data = cfvs_data[d][0].view({-1, terminal_equity.river_hand_abstract_count});
            terminal_equity.fold_value_next_street(_ranges_data, _cfvs_data);
            cfvs_data[d][0].slice(4, 0, 1, 1) *= fold_mutliplier;
            cfvs_data[d][0].slice(4, 1, max_size, 1) *= -fold_mutliplier;
        }
    }
}

void Lookahead::_compute_terminal_equities_next_street_box(const int _iter) {
    assert (tree->street <= 3);

    if (num_pot_sizes == 0)
        return;

    for (int d=1; d<depth+1; ++d) {
        if (d > 1 || first_call_transition) {
            if (ranges_data[d][1].sizes()[0] > 1 || (d == 1 && first_call_transition)) {
                auto parent_indices = std::pair<int, int>(0, -1);
                if (d == 1)
                    parent_indices = std::pair<int, int>(0, 1);
                next_street_boxes_outputs.slice(0, indices[d].first, indices[d].second, 1).copy_(
                        ranges_data[d][1].slice(0, parent_indices.first, parent_indices.second, 1).view({-1, constants.players_count, hand_count}));
            }
        }
    }

    if (tree->current_player== 0)
        next_street_boxes_inputs.copy_(next_street_boxes_outputs);
    else {
        next_street_boxes_inputs.slice(1, 0, 1, 1).copy_(next_street_boxes_outputs.slice(1, 1, 2, 1));
        next_street_boxes_inputs.slice(1, 1, 2, 1).copy_(next_street_boxes_outputs.slice(1, 0, 1, 1));
    }

    if (tree->street == 1) {
//        std::cout << _iter << std::endl;
        if (_iter >= cfr_skip_iters[1]) {
            next_street_boxes_inputs_memory[_iter - cfr_skip_iters[1]].copy_(next_street_boxes_inputs);
//            next_street_boxes->get_value_last_20(next_street_boxes_inputs, next_street_boxes_outputs);
            next_street_boxes->get_value_aux(next_street_boxes_inputs, next_street_boxes_outputs, next_board_idx);
        } else
            next_street_boxes->get_value_aux(next_street_boxes_inputs, next_street_boxes_outputs, next_board_idx);
    }
    else
        next_street_boxes->get_value(next_street_boxes_inputs, next_street_boxes_outputs);

    if (tree->current_player == 0) {
        next_street_boxes_inputs.copy_(next_street_boxes_outputs);
        next_street_boxes_outputs.slice(1, 0, 1, 1).copy_(next_street_boxes_inputs.slice(1, 1, 2, 1));
        next_street_boxes_outputs.slice(1, 1, 2, 1).copy_(next_street_boxes_inputs.slice(1, 0, 1, 1));
    }

    for (int d=1; d<depth+1; ++d) {
        if (d > 1 || first_call_transition) {
            if (ranges_data[d][1].sizes()[0] > 1 || (d == 1 && first_call_transition)) {
                auto parent_indices = std::pair<int, int>(0, -1);
                if (d == 1)
                    parent_indices = std::pair<int, int>(0, 1);
                cfvs_data[d][1].slice(0, parent_indices.first, parent_indices.second, 1).copy_(
                        next_street_boxes_outputs.slice(0, indices[d].first, indices[d].second, 1).view(
                                ranges_data[d][1].slice(0, parent_indices.first, parent_indices.second, 1).sizes()));
            }
        }
    }
}

void Lookahead::_compute_terminal_equities_next_street_resolve(const int _iter) {
    river_lookahead->_compute_terminal_equities_next_street();
    river_lookahead->_compute_cfvs_next_street();

    for (auto& t : next_street_lookahead) {
        auto layer = std::get<0>(t);
        auto action_id = std::get<1>(t);
        auto parent_id = std::get<2>(t);
        auto gp_id = std::get<3>(t);
        auto i = std::get<4>(t);
        auto pot = std::get<5>(t);

        river_lookahead->cfvs_data_hand[i].copy_(
                torch::bmm(river_lookahead->cfvs_data[0][0][0][0][i], terminal_equity.river_hand_abstract.transpose(1,2)));
        if (tree->current_player == 0)
            cfvs_data[layer][action_id][parent_id][gp_id].copy_(river_lookahead->cfvs_data_hand[i].sum(0) / pot / 44);
        else {
            torch::Tensor cfvs_data_hand_sum = river_lookahead->cfvs_data_hand[i].sum(0);
            cfvs_data[layer][action_id][parent_id][gp_id][0].copy_(cfvs_data_hand_sum[1] / pot / 44);
            cfvs_data[layer][action_id][parent_id][gp_id][1].copy_(cfvs_data_hand_sum[0] / pot / 44);
        }
        if (_iter >= cfr_skip_iters[3]) {
            river_lookahead->ranges_data_hand_memory += river_lookahead->ranges_data_hand;
            river_lookahead->cfvs_data_hand_memory += river_lookahead->cfvs_data_hand;
        }
    }
}

void Lookahead::_compute_terminal_equities(const int _iter) {
    if (tree->street == 3 && *std::max_element(tree->bets, tree->bets+2) < params::stack)
        _compute_terminal_equities_next_street_resolve(_iter);
    else if (tree->street < 3)
        _compute_terminal_equities_next_street_box(_iter);

    _compute_terminal_equities_terminal_equity();

    for (int d=1; d<depth+1; ++d)
        cfvs_data[d] *= pot_size[d];
}

void Lookahead::_compute_terminal_equities_next_street() {

    _compute_terminal_equities_terminal_equity_next_street();

    for (int d=1; d<depth+1; ++d)
        cfvs_data[d] *= pot_size[d];
}

void Lookahead::_compute_cfvs() {
    for (int d=depth; d>0; --d) {
        int gp_layer_terminal_actions_count = terminal_actions_count[d-2];
        int ggp_layer_nonallin_bets_count = nonallinbets_count[d-3];

        cfvs_data[d].slice(3, 0, 1, 1) *= empty_action_mask[d].unsqueeze(3);
        cfvs_data[d].slice(3, 1, max_size, 1) *= empty_action_mask[d].unsqueeze(3);
        placeholder_data[d].copy_(cfvs_data[d]);

        placeholder_data[d].slice(3, acting_player[d], acting_player[d]+1, 1) *= current_strategy_data[d].unsqueeze(3);
        regrets_sum[d].copy_(placeholder_data[d].sum(0));

        torch::Tensor swap = swap_data[d-1];
        swap.copy_(regrets_sum[d].view(swap.sizes()));
        cfvs_data[d-1].slice(0, gp_layer_terminal_actions_count, max_size, 1).slice(1, 0, ggp_layer_nonallin_bets_count, 1).copy_(swap.transpose(1,2));
    }
}

void Lookahead::_compute_cfvs_next_street() {
    for (int d=depth; d>0; --d) {
        int gp_layer_terminal_actions_count = terminal_actions_count[d-2];
        int ggp_layer_nonallin_bets_count = nonallinbets_count[d-3];

        cfvs_data[d].slice(5, 0, 1, 1) *= empty_action_mask[d].unsqueeze(5);
        cfvs_data[d].slice(5, 1, max_size, 1) *= empty_action_mask[d].unsqueeze(5);
        placeholder_data[d].copy_(cfvs_data[d]);

        placeholder_data[d].slice(5, acting_player[d], acting_player[d]+1, 1) *= current_strategy_data[d].unsqueeze(5);
        regrets_sum[d].copy_(placeholder_data[d].sum(0));

        if (idx_range_by_depth.find(d) != idx_range_by_depth.end()) {
            torch::Tensor swap = swap_data[d-1].slice(3, 0, idx_range_by_depth[d], 1);
            swap.copy_(regrets_sum[d].view(swap.sizes()));
            cfvs_data[d-1].slice(0, gp_layer_terminal_actions_count, max_size, 1).slice(
                    1, 0, ggp_layer_nonallin_bets_count, 1).slice(3, 0, idx_range_by_depth[d], 1).copy_(swap.transpose(1,2));
        }
        else {
            torch::Tensor swap = swap_data[d-1];
            swap.copy_(regrets_sum[d].view(swap.sizes()));
            cfvs_data[d-1].slice(0, gp_layer_terminal_actions_count, max_size, 1).slice(
                    1, 0, ggp_layer_nonallin_bets_count, 1).copy_(swap.transpose(1,2));
        }
    }
}

void Lookahead::_compute_cumulate_average_cfvs(const int _iter) {
    if (_iter >= cfr_skip_iters[tree->street]) {
        average_cfvs_data[0] += cfvs_data[0];
        average_cfvs_data[1] += cfvs_data[1];
    }
}

void Lookahead::_compute_normalize_average_strategies() {

    torch::Tensor player_avg_strategy = average_strategies_data[1];
    torch::Tensor tmp = player_avg_strategy.sum(0).expand_as(player_avg_strategy);
    player_avg_strategy /= tmp;

    player_avg_strategy.masked_fill_(tmp.eq(0), 0);
    player_avg_strategy[0].masked_fill_(tmp[0].eq(0), 1);
}

void Lookahead::_compute_normalize_average_cfvs() {
    average_cfvs_data[0] /= (cfr_iters[tree->street] - cfr_skip_iters[tree->street]);
}

void Lookahead::_compute_regrets() {
    for (int d=depth; d>0; --d) {
        int gp_layer_terminal_actions_count = terminal_actions_count[d - 2];
        int gp_layer_bets_count = bets_count[d - 2];
        int ggp_layer_nonallin_bets_count = nonallinbets_count[d - 3];

        torch::Tensor current_regrets = current_regrets_data[d];
        current_regrets.copy_(cfvs_data[d].slice(3, acting_player[d], acting_player[d]+1, 1).squeeze(3));

        torch::Tensor next_level_cfvs = cfvs_data[d - 1];

        torch::Tensor parent_inner_nodes = inner_nodes_p1[d - 1];
        parent_inner_nodes.copy_(next_level_cfvs.slice(0, gp_layer_terminal_actions_count, max_size, 1).slice(1, 0, ggp_layer_nonallin_bets_count, 1).slice(
                3, acting_player[d], acting_player[d]+1, 1).squeeze(3).transpose(1, 2).contiguous().view(parent_inner_nodes.sizes()));
        current_regrets -= parent_inner_nodes.view({1, gp_layer_bets_count, -1, hand_count}).expand_as(current_regrets);

        regrets_data[d] += current_regrets;

        regrets_data[d].clamp_(0, max_number);
    }
    if (tree->street == 3 && *std::max_element(tree->bets, tree->bets+2) < params::stack)
        river_lookahead->_compute_regrets_next_street();
}

void Lookahead::_compute_regrets_next_street() {
    for (int d=depth; d>0; --d) {
        int gp_layer_terminal_actions_count = terminal_actions_count[d-2];
        int gp_layer_bets_count = bets_count[d-2];
        int ggp_layer_nonallin_bets_count = nonallinbets_count[d-3];

        torch::Tensor current_regrets = current_regrets_data[d];
        current_regrets.copy_(cfvs_data[d].slice(5, acting_player[d], acting_player[d]+1, 1).squeeze(5));

        torch::Tensor next_level_cfvs = cfvs_data[d-1];

        torch::Tensor parent_inner_nodes = inner_nodes_p1[d-1];
        parent_inner_nodes.copy_(next_level_cfvs.slice(0, gp_layer_terminal_actions_count, max_size, 1).slice(1, 0, ggp_layer_nonallin_bets_count, 1).slice(
                5, acting_player[d], acting_player[d]+1, 1).squeeze(5).transpose(1,2).view(parent_inner_nodes.sizes()));
        if (idx_range_by_depth.find(d-1) != idx_range_by_depth.end())
            parent_inner_nodes = parent_inner_nodes.view({1, gp_layer_bets_count, -1, idx_range_by_depth[d-1], boards_count[4], terminal_equity.river_hand_abstract_count});
        else
            parent_inner_nodes = parent_inner_nodes.view({1, gp_layer_bets_count, -1, river_count, boards_count[4], terminal_equity.river_hand_abstract_count});
        if (idx_range_by_depth.find(d) != idx_range_by_depth.end())
            current_regrets -= parent_inner_nodes.slice(3, 0, idx_range_by_depth[d], 1).expand_as(current_regrets);
        else
            current_regrets -= parent_inner_nodes.expand_as(current_regrets);
        regrets_data[d] += current_regrets;
        regrets_data[d].clamp_(0, max_number);
    }
}

void Lookahead::get_results(std::map<std::string, torch::Tensor> &out) {

    out["strategy"] = average_strategies_data[1].view({-1, hand_count}).clone();

    out["achieved_cfvs"] = average_cfvs_data[0].view({constants.players_count, hand_count})[0].clone();

    if (!reconstruction_opponent_cfvs)
        out["root_cfvs"] = average_cfvs_data[0].view({constants.players_count, hand_count})[1].clone();

    out["root_cfvs_both_players"] = average_cfvs_data[0].view({constants.players_count, hand_count}).clone();
    out["root_cfvs_both_players"][1].copy_(average_cfvs_data[0].view({constants.players_count, hand_count})[0]);
    out["root_cfvs_both_players"][0].copy_(average_cfvs_data[0].view({constants.players_count, hand_count})[1]);

    out["children_cfvs"] = average_cfvs_data[1].slice(3, 0, 1, 1).view({-1, hand_count}).clone();

    torch::Tensor scaler = average_strategies_data[1].view({-1, hand_count}).clone();

    torch::Tensor range_mul = ranges_data[0].slice(3, 0, 1, 1).view({1, hand_count}).expand_as(scaler).clone();

    scaler *= range_mul;
    scaler = scaler.sum(1).view({range_mul.sizes()[0], 1}).expand_as(range_mul);
    scaler = scaler * (cfr_iters[tree->street] - cfr_skip_iters[tree->street]);

    out["children_cfvs"] /= scaler;

//    if (reconstruction_opponent_cfvs)
//        out["opponent_range_last_resolve"] = average_opponent_range / (cfr_iters[tree->street] - cfr_skip_iters[tree->street]);
}

void Lookahead::_set_opponent_starting_range(const int _iter) {
    if (reconstruction_opponent_cfvs) {
        reconstruction_gadget->compute_opponent_range(cfvs_data[0][0][0][0][0], _iter);
        ranges_data[0].slice(3, 1, 2, 1).copy_(reconstruction_gadget->input_opponent_range);

//        if (_iter >= cfr_skip_iters[tree->street]) {
//            average_opponent_range += reconstruction_gadget->input_opponent_range / reconstruction_gadget->input_opponent_range.sum();
//        }
    }
}

void Lookahead::get_chance_action_cfv(const int action_index, const int action, Board& board, torch::Tensor& cfv) {
    if (tree->street <= 2) {
        torch::Tensor box_outputs = next_street_boxes_outputs.view({-1, constants.players_count, hand_count});
        int batch_index = action_to_index[action];
        torch::Tensor pot_mult = next_round_pot_sizes[batch_index];
        next_street_boxes->get_value_on_board(board, box_outputs);
        cfv.copy_(box_outputs[batch_index][1-tree->current_player]);
        cfv *= pot_mult;
    }
    else {
        int board_idx = CardTools::get_board_index(board);
        for (auto& t : next_street_lookahead) {
            auto layer = std::get<0>(t);
            auto action_id = std::get<1>(t);
            auto parent_id = std::get<2>(t);
            auto i = std::get<4>(t);

            if ((action_index == 1 && first_call_transition && layer == 1 && action_id == 1) ||
                (action_index != 1 && first_call_transition && layer == 2 && parent_id == action_index - 2) ||
                (!first_call_transition && layer == 2 && parent_id == action_index - 1)) {
                cfv.copy_(river_lookahead->cfvs_data_hand_memory[i][board_idx][tree->current_player] / river_lookahead->ranges_data_hand_memory[
                        i][board_idx][0].sum());
                return;
            }
        }
    }
}
