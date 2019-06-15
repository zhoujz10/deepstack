//
// Created by zhou on 19-6-15.
//

#ifndef DEEPSTACK_CPP_CARD_TO_STRING_CONVERSION_H
#define DEEPSTACK_CPP_CARD_TO_STRING_CONVERSION_H


#include <map>
#include <string>
#include "../Settings/constants.h"
#include "../Settings/game_settings.h"
#include "../Game/board.h"


class CardToString {

public:
    std::map<int, std::string> card_to_string_table;
    std::map<std::string, int> string_to_card_table;

    CardToString();
    int string_to_card(const std::string& card_string);
    void string_to_board(const std::string& card_string, Board& board);
    int string_to_hand(const std::string& card_string);

};


int card_to_suit(int card);

int card_to_rank(int card);

CardToString& get_card_to_string();


#endif //DEEPSTACK_CPP_CARD_TO_STRING_CONVERSION_H
