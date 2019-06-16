//
// Created by zhou on 19-6-3.
//


#include "../Settings/constants.h"
#include "../Settings/game_settings.h"
#include "poker_tree_builder.h"


PokerTreeBuilder::PokerTreeBuilder(bool is_next, int river_count, int *river_pots)
        : is_next(is_next)
        , river_count(river_count)
        , river_pots(river_pots) {}

void PokerTreeBuilder::_fill_additional_attributes(Node& node) {
    node.pot = *std::min_element(node.bets, node.bets+2);
}

//std::vector<Node>* PokerTreeBuilder::_get_children_player_node(Node &parent_node, int depth) {
void PokerTreeBuilder::_get_children_player_node(Node &parent_node, int depth) {

    parent_node.children = new std::vector<Node>;
    auto children = parent_node.children;

//    fold action
    Node fold_node;
//    auto fold_node = *new Node;
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

    children->push_back(fold_node);

//    check action
    if ((parent_node.current_player == constants.players.P1 && (parent_node.bets[0] == parent_node.bets[1]) && parent_node.street != 1) ||
         (parent_node.current_player == constants.players.P2 && (parent_node.bets[0] == ante) &&
         (parent_node.bets[1] == ante / 2 ) && parent_node.street == 1)) {
        Node check_node;
//        auto check_node = *new Node;
        check_node.node_type = constants.node_types.check;
        check_node.terminal = false;
        check_node.current_player = 1 - parent_node.current_player;
        check_node.street = parent_node.street;
        check_node.board = parent_node.board;
        check_node.set_bets(*std::max_element(parent_node.bets, parent_node.bets+2));

        if (is_next) {
            memcpy(check_node.all_river_bets, parent_node.all_river_bets, 20 * 2 * sizeof(int));
            memcpy(check_node.all_river_valid, parent_node.all_river_valid, 20 * sizeof(bool));
        }
        children->push_back(check_node);
    }
//    transition call
    else if (parent_node.street <= 3 &&
             ((parent_node.bets[0] == parent_node.bets[1]) ||
              (parent_node.bets[0] != parent_node.bets[1] && *std::max_element(parent_node.bets, parent_node.bets+2) < stack))
            ) {
        Node chance_node;
//        auto chance_node = *new Node;

        chance_node.terminal = false;
        chance_node.node_type = constants.node_types.chance_node;
        chance_node.street = parent_node.street;
        chance_node.board = parent_node.board;
        chance_node.current_player = constants.players.chance;
        chance_node.set_bets(*std::max_element(parent_node.bets, parent_node.bets + 2));

        if (root->street == 3) {

            chance_node.is_next_street_root = true;
            int i = std::find(root->river_pots, root->river_pots+19, chance_node.bets[0]) - root->river_pots;
//            std::cout << chance_node.bets[0] << ' ' << i << std::endl;
            if (used_pot[i])
                chance_node.river_idx = i + 1;
            else {
                chance_node.river_idx = i;
                used_pot[i] = true;
            }
            int pot = chance_node.bets[0];
            children->push_back(chance_node);
            if (pot == *std::min_element(root->river_pots, root->river_pots+river_count))
//                root->river_tree_node = &children->back();
                root->river_tree_node = new Node(chance_node);
        }
        else {
            chance_node.is_next_street_root = false;
            children->push_back(chance_node);
        }

//        children->push_back(chance_node);

    }
    else {
//        terminal call - either last street or allin
        Node terminal_call_node;
//        auto terminal_call_node = *new Node;
        terminal_call_node.node_type = constants.node_types.terminal_call;
        terminal_call_node.terminal = true;
        terminal_call_node.current_player = 1 - parent_node.current_player;
        terminal_call_node.street = parent_node.street;
        terminal_call_node.board = parent_node.board;
        terminal_call_node.set_bets(*std::max_element(parent_node.bets, parent_node.bets + 2));

        if (is_next) {
            memcpy(terminal_call_node.all_river_bets, parent_node.all_river_bets, 20 * 2 * sizeof(int));
            memcpy(terminal_call_node.all_river_valid, parent_node.all_river_valid, 20 * sizeof(bool));
            for (int i=0; i<river_count; ++i) {
                int max_b = *std::max_element(parent_node.all_river_bets[i], parent_node.all_river_bets[i]+2);
                terminal_call_node.all_river_bets[i][0] = max_b;
                terminal_call_node.all_river_bets[i][1] = max_b;
            }
        }
        children->push_back(terminal_call_node);
    }
//    3.0 bet actions

    std::vector<std::array<int, 2>> possible_bets;
    std::vector<float> used_pot_fractions;
    get_possible_bets(parent_node, root->street, depth, is_next, possible_bets, used_pot_fractions);

    if (!possible_bets.empty()) {
        for (int i=0; i<possible_bets.size(); ++i) {
            Node child;
//            auto child = *new Node;
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
                        min_raise_size = std::max(min_raise_size, ante);
                        min_raise_size = std::min(max_raise_size, min_raise_size);

                        child.all_river_bets[j][parent_node.current_player] = opponent_bet + (int)(pot * used_pot_fractions[i]);
                        if ((int)(pot * used_pot_fractions[i]) >= max_raise_size || (int)(pot * used_pot_fractions[i]) < min_raise_size)
                            child.all_river_valid[j] = false;
                    }
                }
                else {
                    for (int j=0; j<20; ++j)
                        child.all_river_bets[j][parent_node.current_player] = stack;
                    for (int j=0; j<river_count; ++j)
                        if (child.all_river_bets[j][child.current_player] == stack)
                            child.all_river_valid[j] = false;
                }
            }
            children->push_back(child);
        }
    }
//    return children;
}

//std::vector<Node>* PokerTreeBuilder::_get_children_nodes(Node &parent_node, int depth) {
void PokerTreeBuilder::_get_children_nodes(Node &parent_node, int depth) {

    bool chance_node = (parent_node.current_player == constants.players.chance);
//    transition call -> create a chance node
    if (parent_node.terminal)
        parent_node.children = new std::vector<Node>;
//        return new std::vector<Node>;
//    chance node
    else if (chance_node)
        parent_node.children = new std::vector<Node>;
//        return new std::vector<Node>;
//    inner nodes -> handle bet sizes
    else
        _get_children_player_node(parent_node, depth);
//        return _get_children_player_node(parent_node, depth);
}

void PokerTreeBuilder::_build_tree_dfs(Node &current_node, int depth) {
    _fill_additional_attributes(current_node);
//    auto children = _get_children_nodes(current_node, depth);
//    current_node.children = children;

    _get_children_nodes(current_node, depth);
    auto children = current_node.children;


    int _depth = -1;

    current_node.actions = new int[children->size()];
//    std::cout << children->size() << std::endl;
    for (int i=0; i<children->size(); ++i) {
        (*children)[i].parent = &current_node;
        _build_tree_dfs((*children)[i], depth+1);
        _depth = std::max(_depth, (*children)[i].depth);

        if (i == 0)
            current_node.actions[i] = constants.actions.fold;
        else if (i == 1)
            current_node.actions[i] = constants.actions.ccall;
        else {
            current_node.actions[i] = *std::max_element((*children)[i].bets, (*children)[i].bets + 2);
        }
    }
    current_node.depth = _depth + 1;
}

Node* PokerTreeBuilder::build_tree(Node& build_tree_node) {

    auto root_node = new Node(build_tree_node);

    root = root_node;

    if (root->street == 3) {
        int cur_pot = *std::max_element(root->bets, root->bets+2);
        int river_pots_fractions[20];
        if (root->bets[0] == root->bets[1] && root->current_player == constants.players.P1)
            memcpy(river_pots_fractions, river_pots_fractions_05, 20 * sizeof(int));
        else
            memcpy(river_pots_fractions, river_pots_fractions_1, 20 * sizeof(int));
        while (river_count < 20) {
            int river_pot = cur_pot * river_pots_fractions[river_count];
            if (river_pots_fractions[river_count] > 0 && river_pot < stack) {
                root->river_pots[river_count] = river_pot;
                river_count ++;
            }
            else
                break;
        }
        root->river_count = river_count;
    }

    if (is_next) {
        memcpy(root->river_pots, river_pots, river_count * sizeof(int));
        for (int i=0; i<river_count; ++i) {
            root->all_river_bets[i][0] = root->river_pots[i];
            root->all_river_bets[i][1] = root->river_pots[i];
            root->all_river_valid[i] = true;
        }
    }

    _build_tree_dfs(*root, 0);

    if (root->street == 3 && root->river_count > 0) {
        auto next_street_node = *new Node(root->river_tree_node->board);
        next_street_node.board.cards[4] = 0;
        next_street_node.street = 4;
        next_street_node.current_player = constants.players.P1;
        memcpy(next_street_node.bets, root->river_tree_node->bets, 2 * sizeof(int));

//        auto t = new int[20];
//        memcpy(t, river_pots_fractions_05, 20*sizeof(int));
        auto tree_builder = new PokerTreeBuilder(true, river_count, root->river_pots);
//        auto tree_builder = new PokerTreeBuilder(true, river_count, t);
        auto next_street_root = tree_builder->build_tree(next_street_node);
        root->river_tree_node->next_street_root = next_street_root;
    }
    return root_node;
}

