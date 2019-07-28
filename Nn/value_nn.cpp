//
// Created by zhou on 19-6-10.
//


#include "value_nn.h"


ValueNn::ValueNn(int street) : street(street) {
    torch::autograd::GradMode::set_enabled(false);
    switch (street) {
        case 1: {
            module = torch::jit::load(aux_net_file);
            break;
        }
        case 2: {
            module = torch::jit::load(flop_net_file);
            break;
        }
        case 3: {
            module = torch::jit::load(turn_net_file);
            break;
        }
        default:
            break;
    }
    module->to(device);
};

void ValueNn::get_value(torch::Tensor& inputs, torch::Tensor& outputs) {
    torch::autograd::GradMode::set_enabled(false);
    std::vector<torch::jit::IValue> _inputs;
    _inputs.emplace_back(inputs);
    outputs.copy_(module->forward(_inputs).toTensor());
}

ValueNn& get_aux_nn() {
    static ValueNn aux_net(1);
    return aux_net;
}

ValueNn& get_flop_nn() {
    static ValueNn flop_net(2);
    return flop_net;
}

ValueNn& get_turn_nn() {
    static ValueNn turn_net(3);
    return turn_net;
}

