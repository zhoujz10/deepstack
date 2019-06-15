//
// Created by zhou on 19-6-13.
//

#include "continual_resolving.h"


ContinualResolving::ContinualResolving() {
    Board board;
    card_tools.get_uniform_range(board, starting_player_range);
    resolve_first_node();
}

void ContinualResolving::resolve_first_node() {
    int cards[5] = {-1, -1, -1, -1, -1};
    int bets[2] = { ante, ante/2 };
    Node node(cards, bets);
    node.current_player = constants.players.P2;

    torch::Tensor player_range = torch::ones(hand_count, torch::kFloat32).to(device);
    torch::Tensor opponent_range = torch::ones(hand_count, torch::kFloat32).to(device);
    card_tools.get_uniform_range(node.board, player_range);
    card_tools.get_uniform_range(node.board, opponent_range);

    opponent_range_warm_start = opponent_range.clone();
    first_node_resolving = new Resolving();

    first_node_resolving->resolve_first_node(node, player_range, opponent_range);

    starting_cfvs_p2.copy_(first_node_resolving->get_root_cfv());
}

void ContinualResolving::start_new_hand(ptree& state) {
    if (resolving != first_node_resolving) {
//    TODO: finish this
    }
    resolving = nullptr;
    terminal_equity.set_board();

    card_tools.get_uniform_range(Board(), opponent_range_warm_start);

    last_node = nullptr;
    decision_id = 0;
    position = state.get<int>("position");
    hand_id = state.get<int>("hand_id");
}

void ContinualResolving::_resolve_node(Node &node, ptree &state) {
    if (decision_id == 0 && position == constants.players.P2) {
//    the strategy computation for the first decision node has been already set up
        current_player_range.copy_(starting_player_range);
        resolving = first_node_resolving;
    }
    else {
        assert (!node.terminal);
        assert (node.current_player == position);

        _update_invariant(node, state);

        if (resolving != first_node_resolving) {
//        TODO: finish this
        }
        resolving = new Resolving();
        resolving->resolve(node, current_player_range, current_opponent_cfvs_bound, opponent_range_warm_start);
    }
}

void ContinualResolving::_update_invariant(Node& node, ptree& state) {
    if (last_node != nullptr && last_node->street != node.street) {
        assert (last_node->street + 1 == node.street);
        resolving->get_chance_action_cfv(last_bet, node.board, current_opponent_cfvs_bound);
        card_tools.normalize_range(node.board, current_player_range);
    }
    else if (decision_id == 0) {
        assert (position == constants.players.P1);
        assert (node.street == 1);

        current_player_range = starting_player_range.clone();
        current_opponent_cfvs_bound = starting_cfvs_p2.clone();
    }
    else {
        assert (last_node->street == node.street);
    }
}

int ContinualResolving::compute_action(Node& node, ptree& state) {
    _resolve_node(node, state);
    int sampled_bet = _sample_bet(node, state);
    last_bet = sampled_bet;

    decision_id ++;
    last_node = new Node(node);

    return sampled_bet;
}

int ContinualResolving::_sample_bet(Node& node, ptree& state) {
    auto possible_bets = resolving->get_possible_actions();
    int actions_count = possible_bets->size();

    float hand_strategy[actions_count];
    float hand_strategy_cumsum[actions_count];
    float hand_strategy_sum = 0;
    torch::Tensor action_strategy = torch::zeros(hand_count, torch::kFloat32).to(device);

    for (int i=0; i<actions_count; ++i) {
        int action_bet = (*possible_bets)[i];
        resolving->get_action_strategy(action_bet, action_strategy);
        hand_strategy[i] = action_strategy.data<float>()[hand_id];
        hand_strategy_sum += hand_strategy[i];
        hand_strategy_cumsum[i] = hand_strategy_sum;
    }

    assert (1 - hand_strategy_sum < 0.001);

    std::cout << "strategy" << hand_strategy << std::endl;

    float r = dice();
    int sampled_bet = 0;

    int i = 0;
    for (i=0; i<actions_count; ++i) {
        if (hand_strategy_cumsum[i] >= r) {
            sampled_bet = (*possible_bets)[i];
            break;
        }
    }
    std::cout << "playing action that has prob: " << hand_strategy[i] << std::endl;

    resolving->get_action_cfv(sampled_bet, current_opponent_cfvs_bound);

    torch::Tensor strategy = torch::zeros(hand_count, torch::kFloat32).to(device);
    resolving->get_action_strategy(sampled_bet, strategy);
    current_player_range *= strategy;
    card_tools.normalize_range(node.board, current_player_range);

    return sampled_bet;
}


