//
// Created by zhou on 19-6-13.
//

#include "continual_resolving.h"


ContinualResolving::ContinualResolving() {
    Board board;
    card_tools.get_uniform_range(board, starting_player_range);
    resolve_first_node();

    if (params::use_cache)
        _generate_stack_match();
}

void ContinualResolving::resolve_first_node() {
    int cards[5] = {-1, -1, -1, -1, -1};
    int bets[2] = { ante+params::additional_ante, ante/2+params::additional_ante };
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
        delete resolving;
    }
    resolving = nullptr;
    terminal_equity.set_board();

    card_tools.get_uniform_range(Board(), opponent_range_warm_start);

    last_node = nullptr;
    decision_id = 0;
    position = state.get<int>("position");
    hand_id = state.get<int>("hand_id");
    std::vector<int>().swap(bet_sequence);
    match = true;
    prev_match = true;
    rate_resumed = false;
}

void ContinualResolving::_resolve_node(Node &node, ptree &state) {
    if (pokermaster && (decision_id == 0)) {
        ante *= state.get<int>("rate");
        params::additional_ante *= state.get<int>("rate");
        params::stack = state.get<int>("stack");
        rate = 1;
        delete first_node_resolving;
        delete resolving;
        resolve_first_node();
        node.bets[0] *= state.get<int>("rate");
        node.bets[1] *= state.get<int>("rate");
        rate_resumed = true;
    }

    if (decision_id == 0 && position == constants.players.P2) {
//    the strategy computation for the first decision node has been already set up
        is_round_first_move = true;
        current_player_range.copy_(starting_player_range);
        resolving = first_node_resolving;
    }
    else {
        assert (!node.terminal);
        assert (node.current_player == position);

//        std::cout << "before update invariant" << std::endl;
//        try {
//            print(current_player_range.slice(0, 0, 20, 1));
//        }
//        catch (int e) {
//
//        }

        _update_invariant(node, state);
//        std::cout << "after update invariant" << std::endl;
//        print(current_player_range.slice(0, 0, 20, 1));

        if (resolving != first_node_resolving) {
            delete resolving;
        }
        if (pokermaster && !rate_resumed && state.get<bool>("need_rate_resume")) {
            ante *= state.get<int>("rate");
            params::additional_ante *= state.get<int>("rate");
            params::stack = state.get<int>("stack");
            rate_resumed = true;
            current_opponent_cfvs_bound *= state.get<int>("rate");
        }



//        std::cout << "-------------------------------------" << std::endl;
//        std::cout << current_player_range.slice(0, 500, 510, 1) << std::endl;
//        std::cout << current_opponent_cfvs_bound.slice(0, 500, 510, 1) << std::endl;
//        std::cout << "-------------------------------------" << std::endl;



        resolving = new Resolving();
//        print(current_player_range.slice(0, 0, 20, 1));
//        print(current_opponent_cfvs_bound.slice(0, 0, 20, 1));
//        print(opponent_range_warm_start.slice(0, 0, 20, 1));
        resolving->resolve(node, current_player_range, current_opponent_cfvs_bound, opponent_range_warm_start);
        opponent_range_warm_start.copy_(resolving->resolve_results["opponent_range_last_resolve"]);

//        std::cout << "after resolve" << std::endl;
//        print(current_player_range.slice(0, 0, 20, 1));
//        print(current_opponent_cfvs_bound.slice(0, 0, 20, 1));
    }
}

void ContinualResolving::_update_invariant(Node& node, ptree& state) {
    if (last_node != nullptr && last_node->street != node.street) {
        is_round_first_move = true;
        assert (last_node->street + 1 == node.street);
        resolving->get_chance_action_cfv(last_bet, node.board, current_opponent_cfvs_bound);
        card_tools.normalize_range(node.board, current_player_range);
    }
    else if (decision_id == 0) {
        is_round_first_move = true;
        assert (position == constants.players.P1);
        assert (node.street == 1);

        current_player_range = starting_player_range.clone();
        current_opponent_cfvs_bound = starting_cfvs_p2.clone();
    }
    else {
        is_round_first_move = false;
        assert (last_node->street == node.street);
    }
}

int ContinualResolving::compute_action(Node& node, ptree& state) {

    int sampled_bet;

    if (params::use_cache) {
        if (_load_preflop_cache(node, state)) {
            std::cout << "-----------match-----------" << std::endl;
            if (node.current_player == 0)
                usleep(500000);
            last_node = new Node();
            sampled_bet = _sample_bet_from_cache();
            last_bet = sampled_bet;
            bet_sequence.push_back(sampled_bet);
        }
        else {
            if (prev_match && (decision_id > 0)) {
                std::cout << "--------prev match---------" << std::endl;
//                std::cout << match << ' ' << prev_match << std::endl;
                _resolve_node_cache(node, state);
            }
            else {
                std::cout << "---------not match---------" << std::endl;
                _resolve_node(node, state);
            }
            sampled_bet = _sample_bet(node, state);
            last_bet = sampled_bet;
        }
    }
    else {
        std::cout << "-----not using cache-----" << std::endl;
        _resolve_node(node, state);
        sampled_bet = _sample_bet(node, state);
        last_bet = sampled_bet;
    }

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
        hand_strategy[i] = (float)action_strategy[hand_id].item<float>();
        hand_strategy_sum += hand_strategy[i];
        hand_strategy_cumsum[i] = hand_strategy_sum;
    }

    assert (1 - hand_strategy_sum < 0.001);

    std::cout << "strategy: ";
    for (int i=0; i<actions_count; ++i)
        std::cout << hand_strategy[i] << ' ';
    std::cout << endl;

    float r = dice();
    int sampled_bet = 0;

    int i = 0;
    for (i=0; i<actions_count; ++i) {
        if (hand_strategy_cumsum[i] >= r) {
            sampled_bet = (*possible_bets)[i];
            break;
        }
//        if ((*possible_bets)[i] == aaa[ii]) {
//            sampled_bet = aaa[ii];
//            ii ++;
//            break;
//        }
    }
    std::cout << "playing action that has prob: " << hand_strategy[i] << std::endl;

    resolving->get_action_cfv(sampled_bet, current_opponent_cfvs_bound);

    torch::Tensor strategy = torch::zeros(hand_count, torch::kFloat32).to(device);
    resolving->get_action_strategy(sampled_bet, strategy);
//    std::cout << "strategy" << std::endl;
//    print(strategy.slice(0, 0, 20, 1));




//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;
//    std::cout << current_player_range.slice(0, 30, 50, 1) << std::endl;
//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;


    current_player_range *= strategy;
    card_tools.normalize_range(node.board, current_player_range);


//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;
//    std::cout << strategy.slice(0, 30, 50, 1) << std::endl;
//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;



//    std::cout << resolving->resolve_results["strategy"][0].slice(0, 0, 10, 1) << std::endl;
//    std::cout << resolving->resolve_results["strategy"][1].slice(0, 0, 10, 1) << std::endl;
//    std::cout << resolving->resolve_results["strategy"][2].slice(0, 0, 10, 1) << std::endl;


//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;



//    std::cout << "*************************************" << std::endl;
//    std::cout << current_player_range.slice(0, 0, 10, 1) << std::endl;
//    std::cout << "*************************************" << std::endl;


    return sampled_bet;
}

bool ContinualResolving::_load_preflop_cache(Node& node, ptree& state) {
    prev_match = match;
//    std::cout << "aaa" << std::endl;
    if (!match || state.get<bool>("need_rate_resume")) {
        match = false;
        return false;
    }
//    std::cout << "bbb" << std::endl;
    if (node.street > 1) {
        match = false;
        return false;
    }
//    std::cout << "ccc" << params::stack << std::endl;
    if  (stack_match_list.find(params::stack) == stack_match_list.end()) {
        match = false;
        return false;
    }
//    std::cout << "ddd" << std::endl;
    match_stack = stack_match_list[params::stack];
    char cache_file[200];
    if (position != 1 || decision_id != 0)
        bet_sequence.push_back(*std::max_element(node.bets, node.bets+2));
    _generate_file_name(cache_file);
    if (cache_file[0] == '\0') {
        match = false;
        return false;
    }
//    std::cout << "eee" << std::endl;
    _load_cache_file(cache_file);
    return true;
}

void ContinualResolving::_load_cache_file(const char *s) {
    ifstream f_read(s, ios::binary);
    f_read.read( reinterpret_cast<char *>(&num_actions),sizeof(int32_t) );
    f_read.read( reinterpret_cast<char *>(&num_pot_sizes),sizeof(int32_t) );

    auto *actions = new int32_t[num_actions];
    f_read.read( reinterpret_cast<char *>(&actions[0]), (uint64_t)num_actions*sizeof(int32_t) );
    std::vector<int>().swap(*possible_bets_cache);
    for (int i=0; i<num_actions; ++i) {
        possible_bets_cache->push_back(actions[i]);
//        std::cout << actions[i] << std::endl;
    }

    if (num_pot_sizes > 0) {
        auto *pot_sizes = new int32_t[num_pot_sizes];
        f_read.read( reinterpret_cast<char *>(&pot_sizes[0]), (uint64_t)num_pot_sizes*sizeof(int32_t) );
        pot_sizes_cache = torch::from_blob(pot_sizes, {num_pot_sizes}, torch::kInt32).to(torch::kFloat32).to(device);

        auto *inputs_memory = new float[(cfr_iters[1] - cfr_skip_iters[1]) * num_pot_sizes * player_count * hand_count];
        f_read.read( reinterpret_cast<char *>(&inputs_memory[0]), (uint64_t)(cfr_iters[1] - cfr_skip_iters[1]) * num_pot_sizes * player_count * hand_count * sizeof(float));
        inputs_memory_cache = torch::from_blob(inputs_memory, {cfr_iters[1] - cfr_skip_iters[1], num_pot_sizes, player_count, hand_count}, torch::kFloat32).to(device);
    }

    auto *_current_player_range = new float[hand_count];
    f_read.read( reinterpret_cast<char *>(&_current_player_range[0]), hand_count*sizeof(float) );
    current_player_range = torch::from_blob(_current_player_range, {hand_count}, torch::kFloat32).to(device);

    auto *strategy = new float[num_actions * hand_count];
    f_read.read( reinterpret_cast<char *>(&strategy[0]), num_actions*hand_count*sizeof(float) );
    strategy_cache = torch::from_blob(strategy, {num_actions, hand_count}, torch::kFloat32).to(device);

    auto *children_cfvs = new float[num_actions * hand_count];
    f_read.read( reinterpret_cast<char *>(&children_cfvs[0]), num_actions*hand_count*sizeof(float) );
    children_cfvs_cache = torch::from_blob(children_cfvs, {num_actions, hand_count}, torch::kFloat32).to(device);

    f_read.close();

    delete[] actions;
    delete[] _current_player_range;
    delete[] strategy;
    delete[] children_cfvs;
}

void ContinualResolving::_generate_stack_match() {

    for (int stack=0; stack<1201; ++stack) {
        for ( int ds : { 0, -1, 1, -2, 2 } ) {
            char stack_path[100];
            sprintf(stack_path, "%spreflop_cache_%d/", preflop_cache_root_file.c_str(), stack+ds);
            if (access(stack_path, 0) == 0) {
                stack_match_list[stack] = stack+ds;
                break;
            }
        }
    }
}

int ContinualResolving::_sample_bet_from_cache() {
    int actions_count = possible_bets_cache->size();

    float hand_strategy[actions_count];
    float hand_strategy_cumsum[actions_count];
    float hand_strategy_sum = 0;

    for (int i=0; i<actions_count; ++i) {
        hand_strategy[i] = strategy_cache[i][hand_id].item<float>();
        hand_strategy_sum += hand_strategy[i];
        hand_strategy_cumsum[i] = hand_strategy_sum;
    }

    assert (1 - hand_strategy_sum < 0.001);

    std::cout << "strategy: ";
    for (int i=0; i<actions_count; ++i)
        std::cout << hand_strategy[i] << ' ';
    std::cout << endl;

    float r = dice();
    int sampled_bet = 0;

    for (int i=0; i<actions_count; ++i) {
        if (hand_strategy_cumsum[i] >= r) {
            sampled_bet = (*possible_bets_cache)[i];
            sampled_action_id_cache = i;
            break;
        }
    }
    std::cout << "playing action that has prob: " << hand_strategy[sampled_action_id_cache] << std::endl;

    current_opponent_cfvs_bound.copy_(children_cfvs_cache[sampled_action_id_cache]);


//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;
//    std::cout << current_player_range.slice(0, 30, 50, 1) << std::endl;
//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;


    current_player_range *= strategy_cache[sampled_action_id_cache];
    card_tools.normalize_range(Board(), current_player_range);


//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;
//    std::cout << sampled_action_id_cache << std::endl;
//    std::cout << strategy_cache[sampled_action_id_cache].slice(0, 30, 50, 1) << std::endl;
//    std::cout << "+++++++++++++++++++++++++++++++++++++" << std::endl;



//    std::cout << "*************************************" << std::endl;
//    std::cout << current_player_range.slice(0, 0, 10, 1) << std::endl;
//    std::cout << "*************************************" << std::endl;

    return sampled_bet;
}

void ContinualResolving::_resolve_node_cache(Node& node, ptree &state) {
    if (decision_id == 0 && position == constants.players.P2) {
        is_round_first_move = true;
        current_player_range = starting_player_range.clone();
        resolving = first_node_resolving;
    }
    else {
        _update_invariant_cache(node);
        delete resolving;
        if (pokermaster && !rate_resumed && state.get<bool>("need_rate_resume")) {
            ante *= state.get<int>("rate");
            params::additional_ante *= state.get<int>("rate");
            params::stack = state.get<int>("stack");
            rate_resumed = true;
            current_opponent_cfvs_bound *= state.get<int>("rate");
        }

//        std::cout << "-------------------------------------" << std::endl;
//        std::cout << current_player_range.slice(0, 500, 510, 1) << std::endl;
//        std::cout << current_opponent_cfvs_bound.slice(0, 500, 510, 1) << std::endl;
//        std::cout << "-------------------------------------" << std::endl;

        resolving = new Resolving();
        resolving->resolve(node, current_player_range, current_opponent_cfvs_bound, opponent_range_warm_start);
    }
}

void ContinualResolving::_update_invariant_cache(Node& node) {
    if (last_node && last_node->street != node.street) {
        is_round_first_move = true;
        assert (last_node->street + 1 == node.street);
        int board_idx = CardTools::get_board_index(node.board);
        auto next_street_boxes = get_flop_value();
        next_street_boxes.start_computation(pot_sizes_cache);
        next_street_boxes.init_var();
        next_street_boxes.iter = cfr_skip_iters[1] - 1;
        torch::Tensor box_outputs = torch::zeros({num_pot_sizes, constants.players_count, hand_count}, torch::kFloat32).to(device);
        for (int i=0; i<cfr_iters[1] - cfr_skip_iters[1]; ++i) {
            torch::Tensor inputs_memory_cache_slice = inputs_memory_cache[i];
            next_street_boxes.get_value_aux(inputs_memory_cache_slice, box_outputs, board_idx);
        }
        int batch_index = sampled_action_id_cache - 1;
        torch::Tensor pot_mult = pot_sizes_cache[batch_index];
        next_street_boxes.get_value_on_board(node.board, box_outputs);
//        print(box_outputs[0][0]);
        current_opponent_cfvs_bound = box_outputs[batch_index][1-position];
        current_opponent_cfvs_bound *= pot_mult;
        card_tools.normalize_range(node.board, current_player_range);
    }
    else {
        is_round_first_move = false;
        assert (last_node->street == node.street);
    }
}

void ContinualResolving::_generate_file_name(char *s) {

    std::string hand_string(preflop_cache_root_file);
    hand_string.append("preflop_cache_");
    hand_string.append(std::to_string(match_stack));
    hand_string.append("/");
    if ((position == 1 && bet_sequence.empty()) || (position == 0 && bet_sequence.size() == 1))
        hand_string.append("any");
    else {
        int hand_id_cache = abstraction2hand[street_1_abstraction[hand_id]];
        hand_string.append(card_to_string.hand_to_string(hand_id_cache));
    }
    hand_string.append("_");
    hand_string.append(std::to_string(match_stack));
    hand_string.append("_");
    hand_string.append(std::to_string(position));
    hand_string.append("_");
    auto tostr = static_cast<std::string(*)(int)>(std::to_string);
    hand_string.append(boost::algorithm::join(bet_sequence | boost::adaptors::transformed(tostr), ","));
    std::string s_0 = hand_string + "_0.bin";
    std::string s_1 = hand_string + "_1.bin";
//    std::cout << s_0 << std::endl << s_1 << std::endl;
    if (access(s_0.c_str(), 0) == 0)
        strcpy(s, s_0.c_str());
    else if (access(s_1.c_str(), 0) == 0)
        strcpy(s, s_1.c_str());
    else
        s[0] = '\0';
}

