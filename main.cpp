#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include "terminal_equity.h"
#include "board.h"
#include "card_tools.h"


int main() {
//    auto device = c10::Device(c10::DeviceType::CUDA);

    torch::zeros({2, 2}, torch::kFloat32);
    float a[][3] = {{4,3,2},{1,5,6}};
    torch::Tensor tensor = torch::from_blob(a, {2, 3}, torch::kFloat32);

    tensor = tensor.slice(1, 0, 2);
    std::cout << tensor << std::endl;


//    tensor[tensor > 0].fill_(1);
//    std::cout << tensor << std::endl;

//    tensor = tensor.toType(torch::kByte);
    tensor.masked_fill_(tensor>2, 0);

    char *p = "adsasdsa%djdksjksdjksd%dsdfsffs";

    char *x = p % 3;

    std::cout << tensor << std::endl;

//
//    tensor.fill_(1);
//    std::cout << tensor << std::endl;

//    cout << CardTools::hand_collide[0][0];

//    auto terminal_equity = TerminalEquity();

//    torch::Tensor tensor = torch::rand(2);
//    std::cout << tensor.data<float>()[1] << std::endl;
//    std::cout << (tensor[0].to(torch::kFloat32) > 0) << std::endl;


}