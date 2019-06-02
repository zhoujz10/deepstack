//
// Created by zhou on 19-6-2.
//

#ifndef DEEPSTACK_CPP_NODE_H
#define DEEPSTACK_CPP_NODE_H


#include <vector>
#include "board.h"

struct Node {
    int node_type = 0;

    bool terminal = false;

    int street = 1;

    int bets[2] = { 0, 0 };

    int pot = 0;

    int current_player = 0;

    int depth = -1;

    bool limit_to_street = false;

    bool is_next_street_root = false;

    int river_idx = -1;

    int *actions = nullptr;

    Node *river_tree_node = nullptr;

    Node *parent = nullptr;

    vector<Node>* children;

    int river_pots[20] = { 0 };

    int all_river_bets[20][2] = { { 0, 0 } };

    bool all_river_valid[20] = { false };

    Board board;

};


#endif //DEEPSTACK_CPP_NODE_H
