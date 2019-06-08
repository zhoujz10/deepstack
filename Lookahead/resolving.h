//
// Created by zhou on 19-6-7.
//

#ifndef DEEPSTACK_CPP_RESOLVING_H
#define DEEPSTACK_CPP_RESOLVING_H


#include "../Game/card_tools.h"
#include "lookahead.h"


class Resolving {

public:
    PokerTreeBuilder& tree_builder = *new PokerTreeBuilder();
    Node *lookahead_tree = nullptr;
    Lookahead *lookahead = nullptr;

    void resolve_first_node(Node& node, torch::Tensor& player_range, torch::Tensor& opponent_range, std::map<std::string, torch::Tensor> &results);

private:
    void _create_lookahead_tree(Node& node);
};



#endif //DEEPSTACK_CPP_RESOLVING_H
