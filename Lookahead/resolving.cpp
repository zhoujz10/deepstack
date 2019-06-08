//
// Created by zhou on 19-6-7.
//


#include "resolving.h"


void Resolving::_create_lookahead_tree(Node& node) {
    node.limit_to_street = false;
    lookahead_tree = tree_builder.build_tree(node);
}

void Resolving::resolve_first_node(Node& node, torch::Tensor& player_range, torch::Tensor& opponent_range,
        std::map<std::string, torch::Tensor> &results) {
//    player_range = player_range
//    opponent_range = opponent_range
//    opponent_cfvs = None
    _create_lookahead_tree(node);
    lookahead = new Lookahead();
    lookahead->build_lookahead(*lookahead_tree);
    lookahead->resolve_first_node(player_range, opponent_range);
    lookahead->get_results(results);
}