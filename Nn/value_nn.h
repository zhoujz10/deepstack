//
// Created by zhou on 19-6-10.
//

#ifndef DEEPSTACK_CPP_VALUE_NN_H
#define DEEPSTACK_CPP_VALUE_NN_H


#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <fstream>
#include <iostream>
#include "../Settings/constants.h"


class ValueNn {

public:

    int street;
    int input_size;
    int output_size;
    int middle_size;

    torch::Tensor w0;
    torch::Tensor b0;
    torch::Tensor alpha0;

    torch::Tensor w1;
    torch::Tensor b1;
    torch::Tensor alpha1;

    torch::Tensor w2;
    torch::Tensor b2;
    torch::Tensor alpha2;

    torch::Tensor w3;
    torch::Tensor b3;
    torch::Tensor alpha3;

    torch::Tensor w4;
    torch::Tensor b4;

    torch::Tensor tmp = torch::zeros({2000, 500}, torch::kFloat32).to(device);
    torch::Tensor m = torch::zeros({2000, 500}, torch::kFloat32).to(device);

    //    std::shared_ptr<torch::jit::script::Module> module;
    torch::jit::script::Module module;
    explicit ValueNn(int street);
    void read_weights(const char *s);
    void prelu_block(torch::Tensor& x, torch::Tensor& alpha, torch::Tensor& out);
    void get_value(torch::Tensor& inputs, torch::Tensor& outputs);
};


ValueNn& get_aux_nn();
ValueNn& get_flop_nn();
ValueNn& get_turn_nn();

#endif //DEEPSTACK_CPP_VALUE_NN_H
