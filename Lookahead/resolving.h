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
    std::map<std::string, torch::Tensor> resolve_results;

    ~Resolving();
    void resolve_first_node(Node& node, torch::Tensor& player_range, torch::Tensor& opponent_range);
    void resolve(Node& node, torch::Tensor& player_range, torch::Tensor& opponent_cfvs, torch::Tensor& opponent_range_warm_start);
    vector<int>* get_possible_actions();
    torch::Tensor& get_root_cfv();
    torch::Tensor& get_root_cfv_both_players();
    void get_action_cfv(int action, torch::Tensor& cfv);
    void get_chance_action_cfv(int action, Board& board, torch::Tensor& cfv);
    void get_action_strategy(int action, torch::Tensor& strategy);

private:
    void _create_lookahead_tree(Node& node);
    int _action_to_action_id(int action);
};



#endif //DEEPSTACK_CPP_RESOLVING_H
