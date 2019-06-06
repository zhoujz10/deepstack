//
// Created by zhou on 19-6-3.
//

#ifndef DEEPSTACK_CPP_LOOKAHEAD_H
#define DEEPSTACK_CPP_LOOKAHEAD_H


#include <vector>
#include <map>
#include <tuple>
#include <torch/torch.h>
#include "../Settings/constants.h"
#include "../Settings/game_settings.h"
#include "../TerminalEquity/terminal_equity.h"
#include "../Tree/poker_tree_builder.h"


class LookaheadBuilder;

class Lookahead {
public:

    int ccall_action_index = 0;
    int fold_action_index = 0;
    int num_pot_sizes = 0;
    int depth = 0;
    bool is_next;
    int river_count = 0;

    Node *tree = nullptr;

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

    std::map<int, std::pair<int, int>> term_call_indices;
    int num_term_call_nodes = 0;
    std::map<int, std::pair<int, int>> term_fold_indices;
    int num_term_fold_nodes = 0;

    torch::Tensor ranges_data_call;
    torch::Tensor ranges_data_fold;
    torch::Tensor cfvs_data_call;
    torch::Tensor cfvs_data_fold;

    std::map<int, int> idx_range_by_depth;
    std::map<int, int> parent_action_id;

    std::vector<std::tuple<int, int, int, int, int, int>> next_street_lookahead;
    Lookahead* river_lookahead = nullptr;
    LookaheadBuilder* builder = nullptr;
    std::vector<int> max_depth;

    TerminalEquity& terminal_equity = get_terminal_equity();

    int next_board_idx = 0;

    bool first_call_terminal = false;
    bool first_call_transition = false;
    bool first_call_check = false;

    explicit Lookahead(bool _is_next=false);
    void build_lookahead(Node& _tree);
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


    explicit LookaheadBuilder(Lookahead *ptr);

    void construct_data_structures();
    void set_datastructures_from_tree_dfs(Node& node, const int layer, const int action_id, const int parent_id,
                                          const int gp_id, const int cur_action_id, const int parent_action_id);
    void build_from_tree(Node& tree, const int _river_hand_abstract_count = hand_count);

private:
    void _construct_transition_boxes();
    void _compute_structure();
    void _compute_tree_structures(std::vector<Node*>& current_layer, const int current_depth);
};


#endif //DEEPSTACK_CPP_LOOKAHEAD_H
