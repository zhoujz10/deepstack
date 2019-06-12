//
// Created by zhou on 19-6-10.
//

#ifndef DEEPSTACK_CPP_VALUE_NN_H
#define DEEPSTACK_CPP_VALUE_NN_H


#include <torch/torch.h>
#include <torch/script.h>
#include "../Settings/constants.h"


class ValueNn {

public:

    int street;
    std::shared_ptr<torch::jit::script::Module> module;

    explicit ValueNn(int street);
    void get_value(torch::Tensor& inputs, torch::Tensor& outputs);
};


ValueNn& get_aux_nn();
ValueNn& get_flop_nn();
ValueNn& get_turn_nn();

#endif //DEEPSTACK_CPP_VALUE_NN_H
