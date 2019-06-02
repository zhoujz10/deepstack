//
// Created by zhou on 19-6-2.
//

#ifndef DEEPSTACK_CPP_BET_SIZING_H
#define DEEPSTACK_CPP_BET_SIZING_H

#include <vector>
#include <array>
#include <tuple>
#include "node.h"
#include "constants.h"

vector<float> pot_fractions_by_street[5][2] = {
    { { 0 }, { 0 } },
    { { 0.5, 1 }, { 0.5, 1, 2 } },
    { { 0.5, 1 }, { 1 } },
    { { 0.5, 1 }, { 1 } },
    { { 0.5, 1, 2 }, { 0.5, 1, 2 } },
};

vector<float> default_pot_fractions = { 1 };

auto get_possible_bets(Node& node, int street, int depth, bool is_next) {
    int current_player = node.current_player;
    int opponent = 1 - node.current_player;
    int opponent_bet = node.bets[opponent];

    assert (node.bets[current_player] <= opponent_bet);

//    compute min possible raise size
    int max_raise_size = stack - opponent_bet;
    int min_raise_size = opponent_bet - node.bets[current_player];
    min_raise_size = max(min_raise_size, ante);
    min_raise_size = min(max_raise_size, min_raise_size);

    vector<array<int, 2>> possible_bets;
    vector<float> used_pot_fractions;

    if (min_raise_size == 0)
        return tuple<vector<array<int, 2>>, vector<float>>(possible_bets, used_pot_fractions);
    else if (min_raise_size == max_raise_size) {
        possible_bets.push_back( array<int, 2>( { opponent_bet, opponent_bet } ) );
        possible_bets[0][current_player] = opponent_bet + min_raise_size;
        return tuple<vector<array<int, 2>>, vector<float>>(possible_bets, used_pot_fractions);
    }
    else {
        vector<float>& p_fractions = default_pot_fractions;
        if (!is_next && depth < 2)
            p_fractions = pot_fractions_by_street[street][depth];
//        iterate through all bets and check if they are possible
        auto max_possible_bets_count = p_fractions.size() + 1;  // we can always go allin
        for (int i=0; i<max_possible_bets_count; ++i)
            possible_bets.push_back( array<int, 2>( { opponent_bet, opponent_bet } ) );

//        take pot size after opponent bet is called
        int pot = opponent_bet * 2;
        int used_bets_count = -1;
//        try all pot fractions bet and see if we can use them
        for (auto fraction : p_fractions) {
            int raise_size = (int)(pot * fraction);
            if (min_raise_size <= raise_size < max_raise_size) {
                used_bets_count ++;
                possible_bets[used_bets_count][current_player] = opponent_bet + raise_size;
                used_pot_fractions.push_back(fraction);
            }
        }
//        adding allin
        used_bets_count += 1;
        assert (used_bets_count <= max_possible_bets_count);
        possible_bets[used_bets_count][current_player] = opponent_bet + max_raise_size;
        for (int i=used_bets_count+1; i<max_possible_bets_count; ++i)
            possible_bets.pop_back();
        return tuple<vector<array<int, 2>>, vector<float>>(possible_bets, used_pot_fractions);
    }
}



#endif //DEEPSTACK_CPP_BET_SIZING_H
