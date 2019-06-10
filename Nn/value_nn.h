//
// Created by zhou on 19-6-10.
//

#ifndef DEEPSTACK_CPP_VALUE_NN_H
#define DEEPSTACK_CPP_VALUE_NN_H


#include <torch/torch.h>

class ValueNn {

public:
    void get_value(const torch::Tensor& inputs, torch::Tensor& outputs);
};

ValueNn& get_aux_nn();
ValueNn& get_flop_nn();
ValueNn& get_turn_nn();

#endif //DEEPSTACK_CPP_VALUE_NN_H
