//
// Created by zhou on 19-5-29.
//

#ifndef DEEPSTACK_CPP_CARD_TOOLS_H
#define DEEPSTACK_CPP_CARD_TOOLS_H

#include <torch/torch.h>
#include "constants.h"
#include "game_settings.h"
#include "board.h"
#include "io.h"
#include "runtime.h"


class CardTools {

public:

    static torch::Tensor hand_collide;

    static auto init_hand_collide() {
        auto p = new float[hand_count * hand_count];
        read_pointer(p, hand_collide_file);
        return torch::from_blob(p, {hand_count, hand_count}, torch::kFloat32);
    }

    static void get_possible_hand_indexes(Board& board, torch::Tensor& possible_hand_indexes) {
        possible_hand_indexes.fill_(1);
        if (board.street() == 0)
            return ;
        for (int card : board.cards)
            for (uint16_t hand : card_hand_collide[card])
                possible_hand_indexes[hand] = 0;
    }

    static void get_impossible_hand_indexes(Board& board, torch::Tensor& possible_hand_indexes) {
        get_possible_hand_indexes(board, possible_hand_indexes);
        possible_hand_indexes += -1;
        possible_hand_indexes *= -1;
    }

    static void get_uniform_range(Board &board, torch::Tensor &range) {
        get_possible_hand_indexes(board, range);
        range /= range.sum();
    }

    static int get_boards_count(int street) {
        const int boards_count[5] = { 0, 0, card_count*(card_count-1)*(card_count-2)/6, card_count-3, card_count-4 };
        return boards_count[street];
    }

    static void get_x_round_boards(Board& board, Board* p) {
        int street = board.street() + 1;
        int board_idx = 0;
        switch (street) {
            case 1:
                for (int card_0 = 0; card_0 < card_count; ++card_0) {
                    for (int card_1 = card_0 + 1; card_1 < card_count; ++card_1) {
                        for (int card_2 = card_1 + 1; card_2 < card_count; ++card_2) {
                            p[board_idx].cards[0] = card_0;
                            p[board_idx].cards[1] = card_1;
                            p[board_idx].cards[2] = card_2;
                            board_idx++;
                        }
                    }
                }
            case 2:
                for (int card_3 = 0; card_3 < card_count; ++card_3) {
                    if (board.contain(card_3))
                        continue;
                    p[board_idx].copy_(board);
                    p[board_idx].cards[3] = card_3;
                    board_idx ++;
                }
            case 3:
                for (int card_4 = 0; card_4 < card_count; ++card_4) {
                    if (board.contain(card_4))
                        continue;
                    p[board_idx].copy_(board);
                    p[board_idx].cards[4] = card_4;
                    board_idx ++;
                }
            default:
                return;
        }
    }

    static int get_board_index(Board &board) {
        return get_board_idx_cpp(board.cards[0], board.cards[1], board.cards[2], board.cards[3], board.cards[4]);
    }

    static void normalize_range(Board &board, torch::Tensor &range, torch::Tensor &out) {
        get_possible_hand_indexes(board, out);
        out *= range;
        out /= out.sum();
    }
};


torch::Tensor CardTools::hand_collide = CardTools::init_hand_collide();


#endif //DEEPSTACK_CPP_CARD_TOOLS_H