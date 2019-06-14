//
// Created by zhou on 19-6-13.
//

#ifndef DEEPSTACK_CPP_CONTINUAL_RESOLVING_H
#define DEEPSTACK_CPP_CONTINUAL_RESOLVING_H


#include <torch/torch.h>
#include <boost/property_tree/ptree.hpp>
#include "../Settings/constants.h"
#include "../Game/board.h"
#include "../Game/card_tools.h"
#include "../TerminalEquity/terminal_equity.h"
#include "../Lookahead/resolving.h"


using namespace boost::property_tree;

class ContinualResolving {

public:
    Resolving *resolving = nullptr;
    Resolving *first_node_resolving = nullptr;
    torch::Tensor starting_player_range = torch::ones(hand_count, torch::kFloat32).to(device);
    torch::Tensor current_player_range = torch::ones(hand_count, torch::kFloat32).to(device);
    torch::Tensor current_opponent_cfvs_bound = torch::zeros(hand_count, torch::kFloat32).to(device);
    torch::Tensor starting_cfvs_p2;
    torch::Tensor opponent_range_warm_start;

    Node *last_node = nullptr;
    int decision_id = -1;
    int position = -1;
    int hand_id = -1;

    int last_bet = -10;

    CardTools& card_tools = get_card_tools();
    TerminalEquity& terminal_equity = get_terminal_equity();

    ContinualResolving();
    void resolve_first_node();
    void start_new_hand(ptree& state);
    void compute_action(Node& node, ptree& state);

private:
    void _resolve_node(Node& node, ptree& state);
    void _update_invariant(Node& node, ptree& state);
    void _sample_bet(Node& node, ptree& state);

};


#endif //DEEPSTACK_CPP_CONTINUAL_RESOLVING_H
