//
// Created by zhou on 19-6-2.
//

#ifndef DEEPSTACK_CPP_POKER_TREE_BUILDER_H
#define DEEPSTACK_CPP_POKER_TREE_BUILDER_H


#include <vector>
#include "../Game/board.h"
#include "node.h"
#include "../Game/bet_sizing.h"


class PokerTreeBuilder {
public:

    bool is_next = false;

    int river_count = 0;

    int *river_pots = nullptr;

    bool used_pot[20] = { false };

    Node *root = nullptr;

    explicit PokerTreeBuilder(bool is_next = false, int river_count = 0, int *river_pots = nullptr);

    Node* build_tree(Node& build_tree_node);

private:
    static void _fill_additional_attributes(Node& node);

//    std::vector<Node>* _get_children_player_node(Node &parent_node, int depth);
    void _get_children_player_node(Node &parent_node, int depth);

//    std::vector<Node>* _get_children_nodes(Node &parent_node, int depth);
    void _get_children_nodes(Node &parent_node, int depth);

    void _build_tree_dfs(Node &current_node, int depth);
};


#endif //DEEPSTACK_CPP_POKER_TREE_BUILDER_H
