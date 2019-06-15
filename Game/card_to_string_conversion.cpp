//
// Created by zhou on 19-6-15.
//


#include "card_to_string_conversion.h"


int card_to_suit(const int card) {
    return card % suit_count;
}

int card_to_rank(const int card) {
    return card / suit_count;
}

CardToString::CardToString() {
    for (int card=0; card<card_count; ++card) {
        char rank_name = rank_table[card_to_rank(card)];
        char suit_name = suit_table[card_to_suit(card)];
        std::string card_string = std::string(1, rank_name) + std::string(1, suit_name);
        card_to_string_table[card] = card_string;
        string_to_card_table[card_string] = card;
    }
}

int CardToString::string_to_card(const std::string& card_string) {
    int card = string_to_card_table[card_string];
    return card;
}

void CardToString::string_to_board(const std::string& card_string, Board& board) {
    Board b;
    if (!card_string.empty())
        for (int i=0; i<card_string.length() / 2; ++i)
            b.cards[i] = string_to_card(card_string.substr(2*i, 2));
    board.copy_(b);
}

int CardToString::string_to_hand(const std::string& card_string) {
    assert (card_string.length() == 4);
    int card_0 = string_to_card(card_string.substr(0, 2));
    int card_1 = string_to_card(card_string.substr(2, 2));
    int f_card = std::min(card_0, card_1);
    int s_card = std::max(card_0, card_1);
    int hand_id = 0;
    for (int i=0; i<f_card; ++i)
        hand_id += 51 - i;
    hand_id += s_card - f_card - 1;
    return hand_id;
}

CardToString& get_card_to_string() {
    static CardToString card_to_string;
    return card_to_string;
}