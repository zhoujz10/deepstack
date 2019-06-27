//
// Created by zhou on 19-6-13.
//

#ifndef DEEPSTACK_CPP_CONTINUAL_RESOLVING_H
#define DEEPSTACK_CPP_CONTINUAL_RESOLVING_H


#include <time.h>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include "../Settings/constants.h"
#include "../Game/board.h"
#include "../Game/card_tools.h"
#include "../Game/card_to_string_conversion.h"
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
    torch::Tensor starting_cfvs_p2 = torch::zeros(hand_count, torch::kFloat32).to(device);
    torch::Tensor opponent_range_warm_start;

    std::function<float()> dice = std::bind(distribution, generator);

    Node *last_node = nullptr;
    int decision_id = -1;
    int position = -1;
    int hand_id = -1;

    int last_bet = -10;

    bool match = false;
    bool prev_match = false;

//    int ii = 0;
//    int aaa[9] = {400, 800, -1, -1, -1, 200, 1800, -1, -1};

    std::vector<int> *possible_bets_cache = new std::vector<int>();
    torch::Tensor pot_sizes_cache;
    torch::Tensor inputs_memory_cache;
    torch::Tensor strategy_cache;
    torch::Tensor children_cfvs_cache;
    int match_stack = 0;
    int sampled_action_id_cache = 0;
    int pos_cache = 0;
    int32_t num_actions = 0, num_pot_sizes = 0;
    std::map<int, int> stack_match_list;
    std::vector<int> bet_sequence;


    CardTools& card_tools = get_card_tools();
    CardToString& card_to_string = get_card_to_string();
    TerminalEquity& terminal_equity = get_terminal_equity();

    ContinualResolving();
    void resolve_first_node();
    void start_new_hand(ptree& state);
    int compute_action(Node& node, ptree& state);

private:
    void _resolve_node(Node& node, ptree& state);
    void _update_invariant(Node& node, ptree& state);
    int _sample_bet(Node& node, ptree& state);
    bool _load_preflop_cache(Node& node);
    void _load_cache_file(const char *s);
    int _sample_bet_from_cache();
    void _resolve_node_cache(Node& node);
    void _update_invariant_cache(Node& node);
    void _generate_file_name(char *s);
};



#endif //DEEPSTACK_CPP_CONTINUAL_RESOLVING_H
