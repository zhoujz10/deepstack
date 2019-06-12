//
// Created by zhou on 19-6-12.
//

#ifndef DEEPSTACK_CPP_CFRD_GADGET_H
#define DEEPSTACK_CPP_CFRD_GADGET_H


#include <torch/torch.h>
#include "../Game/board.h"
#include "../Game/card_tools.h"
#include "../Settings/game_settings.h"
#include "../Settings/constants.h"


class CFRDGadget {

public:
    torch::Tensor input_opponent_range = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor input_opponent_value = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor opponent_range_warm_start = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor play_current_strategy = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor terminate_current_strategy = torch::ones({hand_count}, torch::kFloat32).to(device);
    torch::Tensor total_values = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor terminate_regrets = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor play_regrets = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor range_mask = torch::ones({hand_count}, torch::kFloat32).to(device);
    torch::Tensor total_values_p2 = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor play_current_regret = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor terminate_current_regret = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor regret_sum = torch::zeros({hand_count}, torch::kFloat32).to(device);

    CardTools& card_tools = get_card_tools();

    CFRDGadget(Board& board, const torch::Tensor& opponent_cfvs, const torch::Tensor& src_opponent_range_warm_start);
    void compute_opponent_range(const torch::Tensor& current_opponent_cfvs, int iteration);
};

#endif //DEEPSTACK_CPP_CFRD_GADGET_H
