//
// Created by zhou on 19-5-28.
//

#ifndef DEEPSTACK_CPP_BOARD_H
#define DEEPSTACK_CPP_BOARD_H


#include "../Settings/constants.h"

class Board {
public:

    const static size_t board_size = 5;

    int cards[board_size] = {-1, -1, -1, -1, -1};

    Board() = default;

    explicit Board(const int *src) {
        Board board;
        memcpy(board.cards, src, board_size*sizeof(int));
    }

    Board(const int card_0, const int card_1, const int card_2, int const card_3=-1, int const card_4 = -1) {
        cards[0] = card_0;
        cards[1] = card_1;
        cards[2] = card_2;
        cards[3] = card_3;
        cards[4] = card_4;
    }

    Board(const Board&) = default;

    bool is_same(const Board& other) {
        return memcmp(cards, other.cards, board_size*sizeof(int)) == 0;
    }

    Board& copy_(const Board & src) {
        memcpy(cards, src.cards, board_size*sizeof(int));
    }

    int street() {
        if (cards[0] == -1)
            return 1;
        if (cards[3] == -1)
            return 2;
        if (cards[4] == -1)
            return 3;
        return 4;
    };

    friend void print(Board board) {
        std::cout << "{ Cards on board : ";
        for(auto card : board.cards)
            std::cout << card << ' ';
        std::cout << '}' << std::endl;
    }

    bool contain(int card) {
        return cards[0] == card || cards[1] == card || cards[2] == card || cards[3] == card || cards[4] == card;
    }
};


#endif //DEEPSTACK_CPP_BOARD_H
