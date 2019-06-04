//
// Created by zhou on 19-6-3.
//

#ifndef DEEPSTACK_CPP_LOOKAHEAD_H
#define DEEPSTACK_CPP_LOOKAHEAD_H


#include <vector>
#include <map>
#include <torch/torch.h>
#include "constants.h"
#include "game_settings.h"
#include "poker_tree_builder.h"


class Lookahead {
public:

    int ccall_action_index = 0;
    int fold_action_index;
    int num_pot_sizes;
    int depth;
    bool is_next;
    int river_count;

    Node *tree;

    torch::Tensor acting_player;

    std::map<int, int> bets_count;
    std::map<int, int> nonallinbets_count;
    std::map<int, int> terminal_actions_count;
    std::map<int, int> actions_count;

    std::map<int, int> nonterminal_nodes_count;
    std::map<int, int> nonterminal_nonallin_nodes_count;
    std::map<int, int> all_nodes_count;
    std::map<int, int> terminal_nodes_count;
    std::map<int, int> allin_nodes_count;
    std::map<int, int> inner_nodes_count;

    std::map<int, torch::Tensor> pot_size;
    std::map<int, torch::Tensor> ranges_data;
    std::map<int, torch::Tensor> average_strategies_data;
    std::map<int, torch::Tensor> current_strategy_data;
    std::map<int, torch::Tensor> cfvs_data;
    std::map<int, torch::Tensor> average_cfvs_data;
    std::map<int, torch::Tensor> regrets_data;
    std::map<int, torch::Tensor> current_regrets_data;
    std::map<int, torch::Tensor> positive_regrets_data;
    std::map<int, torch::Tensor> placeholder_data;
    std::map<int, torch::Tensor> regrets_sum;
    std::map<int, torch::Tensor> empty_action_mask;
    std::map<int, torch::Tensor> inner_nodes;
    std::map<int, torch::Tensor> inner_nodes_p1;
    std::map<int, torch::Tensor> swap_data;

    torch::Tensor ranges_data_hand;
    torch::Tensor ranges_convert;
    torch::Tensor cfvs_data_hand;
    torch::Tensor cfvs_data_hand_memory;
    torch::Tensor ranges_data_hand_memory;

    std::map<int, int> idx_range_by_depth;

    Lookahead();
    void build_lookahead(PokerTreeBuilder& tree);
    void resolve_first_node(torch::Tensor player_range, torch::Tensor opponent_range);
    void resolve(torch::Tensor player_range, torch::Tensor opponent_cfvs, torch::Tensor opponent_range_warm_start);
    torch::Tensor& get_chance_action_cfv(const int action_index, const int action, Board& board);
    auto get_results();

private:
    void _compute();
    void _compute_current_strategies();
    void _compute_current_strategies_next_street();
    void _compute_ranges();
    void _compute_ranges_next_street();
    void _compute_update_average_strategies(const int _iter);
    void _compute_terminal_equities_terminal_equity();
    void _compute_terminal_equities_terminal_equity_next_street();
    void _compute_terminal_equities_next_street_box(const int _iter);
    void _compute_terminal_equities_next_street_resolve(const int _iter);
    void _compute_terminal_equities(const int _iter);
    void _compute_terminal_equities_next_street();
    void _compute_cfvs();
    void _compute_cfvs_next_street();
    void _compute_cumulate_average_cfvs(const int _iter);
    void _compute_normalize_average_strategies();
    void _compute_normalize_average_cfvs();
    void _compute_regrets();
    void _compute_regrets_next_street();
    void _set_opponent_starting_range(const int _iter);
};

class LookaheadBuilder {
public:

    int river_hand_abstract_count = hand_count;
    Lookahead* lookahead = nullptr;


    explicit LookaheadBuilder(Lookahead& src);

    void construct_data_structures();
    void set_datastructures_from_tree_dfs(Node& node, const int layer, const int action_id, const int parent_id,
                                          const int gp_id, const int cur_action_id, const int parent_action_id);
    void build_from_tree(PokerTreeBuilder& tree, const int river_hand_abstract_count = hand_count);

private:
    void _construct_transition_boxes();
    void _compute_structure();
    void _compute_tree_structures(std::vector<Node&>& layer, const int current_depth);
};


#endif //DEEPSTACK_CPP_LOOKAHEAD_H
