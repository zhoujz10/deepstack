//#include <torch/torch.h>
//#include <ATen/ATen.h>
//#include <iostream>
//#include "terminal_equity.h"
#include "board.h"
#include "node.h"
#include "card_tools.h"
#include "bet_sizing.h"
#include "poker_tree_builder.h"


int main() {
//    auto device = c10::Device(c10::DeviceType::CUDA);

    torch::zeros({2, 2}, torch::kFloat32);
    float a[][3] = {{4,3,2},{1,5,6}};
    torch::Tensor tensor = torch::from_blob(a, {2, 3}, torch::kFloat32);
    std::cout << tensor.sizes() << endl;
    tensor = tensor.slice(1, 0, 2);
    std::cout << tensor << std::endl;

    int cards[5] = { 1, 2, 3, -1, -1 };
    int bets[2] = { 100, 100 };
    Node build_tree_params( cards, bets );
    PokerTreeBuilder ptb;
    auto root_node = ptb.build_tree(build_tree_params);
    for (auto& child : *root_node->children)
        std::cout << child.bets[0] << ' ' << child.bets[1] << std::endl;
    std::cout << (*(*root_node->children)[1].children)[1].children->size() << std::endl;



//    tensor[tensor > 0].fill_(1);
//    std::cout << tensor << std::endl;

//    tensor = tensor.toType(torch::kByte);
//    tensor.masked_fill_(tensor>2, 0);
//    std::cout << tensor << std::endl;

//
//    tensor.fill_(1);
//    std::cout << tensor << std::endl;

//    cout << CardTools::hand_collide[0][0];

//    auto terminal_equity = TerminalEquity();

//    torch::Tensor tensor = torch::rand(2);
//    std::cout << tensor.data<float>()[1] << std::endl;
//    std::cout << (tensor[0].to(torch::kFloat32) > 0) << std::endl;


}