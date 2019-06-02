//
// Created by zhou on 19-6-2.
//

#ifndef DEEPSTACK_CPP_POKER_TREE_BUILDER_H
#define DEEPSTACK_CPP_POKER_TREE_BUILDER_H


#include <vector>
#include "board.h"
#include "node.h"
#include "bet_sizing.h"


class PokerTreeBuilder {
public:

    bool is_next;

    int river_count;

    int *river_pots;

    bool used_pot[20] = { false };

    Node root;

    explicit PokerTreeBuilder(bool _is_next = false, int river_count = 0, int *river_pots = nullptr);

    static void _fill_additional_attributes(Node& node);

    vector<Node> _get_children_player_node(Node &parent_node, int depth);

    vector<Node> _get_children_nodes(Node &parent_node, int depth);

    void _build_tree_dfs(Node &current_node, int depth);
};

PokerTreeBuilder::PokerTreeBuilder(bool is_next, int river_count, int *river_pots)
    : is_next(is_next)
    , river_count(river_count)
    , river_pots(river_pots) {}

void PokerTreeBuilder::_fill_additional_attributes(Node& node) {
    node.pot = *min_element(node.bets, node.bets+1);
}

vector<Node> PokerTreeBuilder::_get_children_player_node(Node &parent_node, int depth) {

    vector<Node> children;

//    fold action
    Node fold_node;
    fold_node.node_type = constants.node_types.terminal_fold;
    fold_node.terminal = true;
    fold_node.current_player = 1 - parent_node.current_player;
    fold_node.street = parent_node.street;
    fold_node.board.copy_(parent_node.board);
    memcpy(fold_node.bets, parent_node.bets, 2 * sizeof(int));

    if (is_next) {
        memcpy(fold_node.all_river_bets, parent_node.all_river_bets, 20 * 2 * sizeof(int));
        memcpy(fold_node.all_river_valid, parent_node.all_river_valid, 20 * sizeof(bool));
    }

    children.push_back(fold_node);

//    check action
    if ((parent_node.current_player == constants.players.P1 && (parent_node.bets[0] == parent_node.bets[1]) && parent_node.street != 1) or
        (parent_node.current_player == constants.players.P2 && (parent_node.bets[0] == ante) &&
         (parent_node.bets[1] == ante / 2 ) && parent_node.street == 1))
    {
        Node check_node;
        check_node.node_type = constants.node_types.check;
        check_node.terminal = false;
        check_node.current_player = 1 - parent_node.current_player;
        check_node.street = parent_node.street;
        check_node.board = parent_node.board;
        memset(check_node.bets, *max_element(parent_node.bets, parent_node.bets+1), 2 * sizeof(int));

        if (is_next) {
            memcpy(check_node.all_river_bets, parent_node.all_river_bets, 20 * 2 * sizeof(int));
            memcpy(check_node.all_river_valid, parent_node.all_river_valid, 20 * sizeof(bool));
        }
        children.push_back(check_node);
    }
//    transition call
    else if (parent_node.street <= 3 &&
         ((parent_node.bets[0] == parent_node.bets[1]) ||
          (parent_node.bets[0] != parent_node.bets[1] && *max_element(parent_node.bets, parent_node.bets+1) < stack))
    ) {
        Node chance_node;
        chance_node.terminal = false;
        chance_node.node_type = constants.node_types.chance_node;
        chance_node.street = parent_node.street;
        chance_node.board = parent_node.board;
        chance_node.current_player = constants.players.chance;
        memset(chance_node.bets, *max_element(parent_node.bets, parent_node.bets + 1), 2 * sizeof(int));

        if (root.street == 3) {

            chance_node.is_next_street_root = true;
            int i = *find(root.river_pots, root.river_pots+19, chance_node.bets[0]);
            if (used_pot[i])
                chance_node.river_idx = i + 1;
            else {
                chance_node.river_idx = i;
                used_pot[i] = true;
            }
            int pot = chance_node.bets[0];
            if (pot == *min_element(root.river_pots, root.river_pots+river_count-1))
                root.river_tree_node = &chance_node;
        }
        else
            chance_node.is_next_street_root = false;

        children.push_back(chance_node);
    }
    else {
//        terminal call - either last street or allin
        Node terminal_call_node;
        terminal_call_node.node_type = constants.node_types.terminal_call;
        terminal_call_node.terminal = true;
        terminal_call_node.current_player = 1 - parent_node.current_player;
        terminal_call_node.street = parent_node.street;
        terminal_call_node.board = parent_node.board;
        memset(terminal_call_node.bets, *max_element(parent_node.bets, parent_node.bets + 1), 2 * sizeof(int));

        if (is_next) {
            memcpy(terminal_call_node.all_river_bets, parent_node.all_river_bets, 20 * 2 * sizeof(int));
            memcpy(terminal_call_node.all_river_valid, parent_node.all_river_valid, 20 * sizeof(bool));
            for (int i=0; i<river_count; ++i) {
                int max_b = *max_element(parent_node.all_river_bets[i], parent_node.all_river_bets[i]+1);
                memset(terminal_call_node.all_river_bets[i], max_b, 2 * sizeof(int));
            }
        }
        children.push_back(terminal_call_node);
    }
//    3.0 bet actions

//    if (parent_node.current_player == constants.players.P2 && (parent_node.bets[0] == ante) && (parent_node.bets[1] == ante / 2) && parent_node.street == 1) {
//        pot_fractions_by_street[1][0] = { 0.75, 1,25 };
//        auto [possible_bets, used_pot_fractions] = get_possible_bets(parent_node, root.street, depth, is_next);
//        pot_fractions_by_street[1][0] = { 0.5, 1 };
//    }
//    else
//        auto [possible_bets, used_pot_fractions] = get_possible_bets(parent_node, root.street, depth, is_next);

//    auto [possible_bets, used_pot_fractions] = get_possible_bets(parent_node, root.street, depth, is_next);
    tuple<vector<array<int, 2>>, vector<float>> result = get_possible_bets(parent_node, root.street, depth, is_next);
    vector<array<int, 2>> possible_bets = get<0>(result);
    vector<float> used_pot_fractions = get<1>(result);

    if (!possible_bets.empty()) {
        for (int i=0; i<possible_bets.size(); ++i) {
            Node child;
            child.terminal = false;
            child.node_type = constants.node_types.inner_node;
            child.parent = &parent_node;
            child.current_player = 1 - parent_node.current_player;
            child.street = parent_node.street;
            child.board = parent_node.board;
            child.bets[0] = possible_bets[i][0];
            child.bets[1] = possible_bets[i][1];

            if (is_next) {
                memcpy(child.all_river_bets, parent_node.all_river_bets, 20 * 2 * sizeof(int));
                memcpy(child.all_river_valid, parent_node.all_river_valid, 20 * sizeof(bool));
                if (i < possible_bets.size() - 1) {
                    for (int j=0; j<river_count; ++j) {
                        int opponent_bet = parent_node.all_river_bets[j][child.current_player];
                        int pot = opponent_bet * 2;
                        int max_raise_size = stack - opponent_bet;
                        int min_raise_size = opponent_bet - parent_node.all_river_bets[j][parent_node.current_player];
                        min_raise_size = max(min_raise_size, ante);
                        min_raise_size = min(max_raise_size, min_raise_size);

                        child.all_river_bets[j][parent_node.current_player] = opponent_bet + (int)(pot * used_pot_fractions[i]);
                        if (pot * used_pot_fractions[i] >= max_raise_size || pot * used_pot_fractions[i] < min_raise_size)
                            child.all_river_valid[j] = 0;
                    }
                }
                else {
                    for (int j=0; j<20; ++j)
                        child.all_river_bets[j][parent_node.current_player] = stack;
                    for (int j=0; j<river_count; ++j)
                        if (child.all_river_bets[j][child.current_player] == stack) {
                            child.all_river_valid[j] = false;
                            children.push_back(child);
                        }
                }
            }
        }
    }
    return children;
}

vector<Node> PokerTreeBuilder::_get_children_nodes(Node &parent_node, int depth) {

    bool chance_node = (parent_node.current_player == constants.players.chance);
//    transition call -> create a chance node
    if (parent_node.terminal)
        return vector<Node>();
//    chance node
    else if (chance_node)
        return vector<Node>();
//    inner nodes -> handle bet sizes
    else
        return _get_children_player_node(parent_node, depth);
}

void PokerTreeBuilder::_build_tree_dfs(Node &current_node, int depth) {
        _fill_additional_attributes(current_node);
        auto children = _get_children_nodes(current_node, depth);
        current_node.children = &children;

        int _depth = -1;

        current_node.actions = new int[children.size()];
        for (int i=0; i<children.size(); ++i) {
            children[i].parent = &current_node;
            _build_tree_dfs(children[i], depth+1);
            _depth = max(_depth, children[i].depth);

            if (i == 0)
                current_node.actions[i] = constants.actions.fold;
            else if (i == 1)
                current_node.actions[i] = constants.actions.ccall;
            else
                current_node.actions[i] = *max_element(children[i].bets, children[i].bets + 1);
        }
        current_node.depth = _depth + 1;
}






#endif //DEEPSTACK_CPP_POKER_TREE_BUILDER_H
