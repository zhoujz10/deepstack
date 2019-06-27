//
// Created by zhou on 19-6-7.
//


#include "resolving.h"


Resolving::~Resolving() {
//    delete &tree_builder;
//    delete lookahead_tree;
//    std::cout << "Resolving released." << std::endl;
    delete lookahead;
//    delete &resolve_results;
}

void Resolving::_create_lookahead_tree(Node& node) {
    node.limit_to_street = false;
    lookahead_tree = tree_builder.build_tree(node);
}

void Resolving::resolve_first_node(Node& node, torch::Tensor& player_range, torch::Tensor& opponent_range) {
    _create_lookahead_tree(node);
    lookahead = new Lookahead();
    lookahead->build_lookahead(*lookahead_tree);
    lookahead->resolve_first_node(player_range, opponent_range);
    lookahead->get_results(resolve_results);
}

void Resolving::resolve(Node& node, torch::Tensor& player_range, torch::Tensor& opponent_cfvs, torch::Tensor& opponent_range_warm_start) {
    _create_lookahead_tree(node);
    lookahead = new Lookahead();
    lookahead->build_lookahead(*lookahead_tree);
    lookahead->resolve(player_range, opponent_cfvs, opponent_range_warm_start);
    lookahead->get_results(resolve_results);
}

int Resolving::_action_to_action_id(const int action) {
    auto actions = get_possible_actions();
    int action_id = -1;
    for (int i=0; i<actions->size(); ++i)
        if (action == (*actions)[i])
            action_id = i;
    assert (action_id != -1);
    return action_id;
}

vector<int>* Resolving::get_possible_actions() {
    auto possible_actions = new vector<int>;
    for (int i=0; i<lookahead_tree->children->size(); ++i)
        possible_actions->push_back(lookahead_tree->actions[i]);
    return possible_actions;
}

torch::Tensor& Resolving::get_root_cfv() {
    return resolve_results["root_cfvs"];
}

torch::Tensor& Resolving::get_root_cfv_both_players() {
    return resolve_results["root_cfvs_both_players"];
}

void Resolving::get_action_cfv(const int action, torch::Tensor& cfv) {
    int action_id = _action_to_action_id(action);
    cfv.copy_(resolve_results["children_cfvs"][action_id]);
}

void Resolving::get_chance_action_cfv(const int action, Board& board, torch::Tensor& cfv) {
    if (board.street() == 2) {
        int board_idx = CardTools::get_board_index(board);
        lookahead->next_board_idx = board_idx;
        lookahead->next_street_boxes->start_computation(lookahead->next_round_pot_sizes);
        lookahead->next_street_boxes->init_var();
        lookahead->next_street_boxes->iter = cfr_skip_iters[1] - 1;
        for (int i=0; i<cfr_iters[1]-cfr_skip_iters[1]; ++i) {
            torch::Tensor next_street_boxes_inputs_memory_iter = lookahead->next_street_boxes_inputs_memory[i];
            lookahead->next_street_boxes->get_value_aux(next_street_boxes_inputs_memory_iter, lookahead->next_street_boxes_outputs, board_idx);
        }
    }
    int action_id = _action_to_action_id(action);
    lookahead->get_chance_action_cfv(action_id, action, board, cfv);
}

void Resolving::get_action_strategy(const int action, torch::Tensor& strategy) {
    int action_id = _action_to_action_id(action);
    strategy.copy_(resolve_results["strategy"][action_id]);
}

