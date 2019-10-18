//
// Created by zhou on 19-5-29.
//

#ifndef DEEPSTACK_CPP_CARD_TOOLS_H
#define DEEPSTACK_CPP_CARD_TOOLS_H

#include <torch/torch.h>
#include "../Settings/constants.h"
#include "../Settings/game_settings.h"
#include "board.h"
#include "../io.h"
#include "../runtime.h"


class CardTools {

public:

    torch::Tensor hand_collide = torch::zeros({hand_count, hand_count}, torch::kFloat32).to(device);

    CardTools() {
        auto p = new float[hand_count * hand_count];
        memset(p, 0, hand_count*hand_count*sizeof(float));
        read_pointer(p, hand_collide_file);
        hand_collide.copy_(torch::from_blob(p, {hand_count, hand_count}, torch::kFloat32));
//        delete[] p;
    }

    static void get_possible_hand_indexes(const Board& board, int *possible_hand_indexes) {
        for (int i=0; i<hand_count; ++i)
            possible_hand_indexes[i] = 1;
        if (board.street() == 1)
            return ;
        for (auto _card : board.cards)
            for (auto hand : card_hand_collide[_card]) {
                if (_card < 0)
                    break;
                possible_hand_indexes[hand] = 0;
            }
    }

    void get_possible_hand_indexes(const Board& board, torch::Tensor& possible_hand_indexes) {
        possible_hand_indexes.fill_(1);
        if (board.street() == 1)
            return ;
        for (auto _card : board.cards)
            for (auto hand : card_hand_collide[_card]) {
                if (_card < 0)
                    break;
                possible_hand_indexes[hand] = 0;
            }
    }

    void get_impossible_hand_indexes(const Board& board, torch::Tensor& possible_hand_indexes) {
        get_possible_hand_indexes(board, possible_hand_indexes);
        possible_hand_indexes += -1;
        possible_hand_indexes *= -1;
    }

    void get_uniform_range(const Board &board, torch::Tensor &range) {
        get_possible_hand_indexes(board, range);
        range /= range.sum();
    }

    static int get_boards_count(int street) {
        return boards_count[street];
    }

    static void get_x_round_boards(Board& board, Board* p) {
        int street = board.street() + 1;
        int board_idx = 0;
        switch (street) {
            case 2: {
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
                break;
            }
            case 3: {
                for (int card_3 = 0; card_3 < card_count; ++card_3) {
                    if (board.contain(card_3))
                        continue;
                    p[board_idx].copy_(board);
                    p[board_idx].cards[3] = card_3;
                    board_idx++;
                }
                break;
            }
            case 4: {
                for (int card_4 = 0; card_4 < card_count; ++card_4) {
                    if (board.contain(card_4))
                        continue;
                    p[board_idx].copy_(board);
                    p[board_idx].cards[4] = card_4;
                    board_idx++;
                }
                break;
            }
            default:
                break;
        }
    }

    static int get_board_index(Board &board) {
        return get_board_idx_cpp(board.cards[0], board.cards[1], board.cards[2], board.cards[3], board.cards[4]);
    }

    void normalize_range(const Board &board, torch::Tensor &range) {
        torch::Tensor tmp = torch::ones(hand_count, torch::kFloat32).to(device);
        get_possible_hand_indexes(board, tmp);
        tmp *= range;
        range.copy_(tmp / tmp.sum());
    }
};


CardTools& get_card_tools();


#endif //DEEPSTACK_CPP_CARD_TOOLS_H
