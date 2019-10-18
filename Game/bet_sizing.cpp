//
// Created by zhou on 19-6-3.
//


#include "../Settings/constants.h"
#include "../Settings/game_settings.h"
#include "bet_sizing.h"


void get_possible_bets(Node& node, int street, int depth, bool is_next,
                       std::vector<std::array<int, 2>>& possible_bets, std::vector<float>& used_pot_fractions) {

    int current_player = node.current_player;
    int opponent = 1 - node.current_player;
    int opponent_bet = node.bets[opponent];

    assert (node.bets[current_player] <= opponent_bet);

//    compute min possible raise size
    int max_raise_size = params::stack - opponent_bet;
    int min_raise_size = opponent_bet - node.bets[current_player];
    min_raise_size = std::max(min_raise_size, ante);
    min_raise_size = std::min(max_raise_size, min_raise_size);

    if (min_raise_size == 0)
        return;

    if (min_raise_size == max_raise_size) {
        possible_bets.push_back( { opponent_bet, opponent_bet } );
        possible_bets[0][current_player] = opponent_bet + min_raise_size;
    }
    else {
        int fraction_count = 0;
        if (node.current_player == constants.players.P2 && (node.bets[0] == ante+params::additional_ante) &&
        (node.bets[1] == ante / 2+params::additional_ante) && node.street == 1 && pokermaster && params::position == 1) {
            if (params::additional_ante == 0) {
                street = 0;
                depth = 1;
            }
            else if (minimum_ante / params::minimum_additional_ante == 4) {  // additional ante = 0.25 * ante
                street = 5;
                depth = 0;
            }
            else {  // additional ante = 0.5 * ante
                street = 5;
                depth = 1;
            }
        }
        else if (depth == 0 && node.street == 3 && ((float)*std::max_element(node.bets, node.bets+2) / params::stack <= 0.03) && pokermaster) {
            street = 0;
            depth = 0;
        }
        else if (is_next || depth >= 2 || generate_mode) {
            street = 0;
            depth = 0;
        }
        else if (pokermaster && street == 4 && depth == 0) {
            street = 6;
        }
        for (int idx=0; idx<3; ++idx) {
            if (pot_fractions_by_street[street][depth][idx] < 0)
                break;
            fraction_count ++;
        }

//        iterate through all bets and check if they are possible
        auto max_possible_bets_count = fraction_count + 1;  // we can always go allin
        for (int i=0; i<max_possible_bets_count; ++i)
            possible_bets.push_back( { opponent_bet, opponent_bet } );

//        take pot size after opponent bet is called
        int pot = opponent_bet * 2;
        int used_bets_count = -1;
//        try all pot fractions bet and see if we can use them
        for (int fraction_idx = 0; fraction_idx < fraction_count; ++fraction_idx) {
            int raise_size = (int)round(pot * pot_fractions_by_street[street][depth][fraction_idx]);
            if (min_raise_size <= raise_size && raise_size < max_raise_size) {
                used_bets_count ++;
                possible_bets[used_bets_count][current_player] = opponent_bet + raise_size;
                used_pot_fractions.push_back(pot_fractions_by_street[street][depth][fraction_idx]);
            }
        }
//        adding allin
        used_bets_count ++;
        for (int i=used_bets_count+1; i<max_possible_bets_count; ++i)
            possible_bets.pop_back();
        assert (used_bets_count <= max_possible_bets_count);
        possible_bets[used_bets_count][current_player] = opponent_bet + max_raise_size;
    }
}
