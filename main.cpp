#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include "TerminalEquity/terminal_equity.h"
#include "Game/board.h"
#include "Tree/node.h"
#include "Game/card_tools.h"
#include "Game/bet_sizing.h"
#include "Tree/poker_tree_builder.h"



int main() {

    clock_t start = clock();

//    auto device = c10::Device(c10::DeviceType::CUDA);


    float a[][3] = {{4,3,2},{1,5,6}};
    torch::Tensor tensor = torch::from_blob(a, {2, 3}, torch::kFloat32);
    std::cout << tensor.sizes() << endl;
//    tensor = tensor.slice(1, 0, 2);
//    std::cout << tensor << std::endl;
//
//    torch::Tensor tensor_2 = torch::zeros({hand_count, hand_count}, torch::kFloat32);
//    CardTools& c = get_card_tools();
//    auto te = get_terminal_equity();

    torch::Tensor t2;

    t2 = tensor.view({1, 6});

    t2[0] = 100;

    std::cout << t2 << endl;
    std::cout << tensor << endl;


    std::cout << tensor.sum(0) << std::endl;
    std::cout << tensor.sum(1) << std::endl;


    int cards[5] = { 1, 2, 3, -1, -1 };
    int bets[2] = { 100, 100 };
    Node build_tree_params( cards, bets );
    PokerTreeBuilder ptb;
    auto root_node = ptb.build_tree(build_tree_params);
    for (auto& child : *root_node->children)
        std::cout << child.bets[0] << ' ' << child.bets[1] << std::endl;
    std::cout << (*(*root_node->children)[1].children)[1].children->size() << std::endl;

    std::cout << ceil(10) << std::endl;
    std::cout << ceil(10.12) << std::endl;

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

//    std::vector<int> tmp = { 5, 4, 3, 6, 7 };
//    for (auto it=tmp.begin(); it!=tmp.end(); ++it)
//        std::cout << it - tmp.begin() << ' ';
//    std::cout << std::endl;
//
//    auto test = torch::zeros({5, 3, 2}, torch::kFloat32);
//    test[1][1] = 1;
//    test[3] = 2;
//    print(test[-1]);
//
//    test.slice(0, 3, -1, 1) = 3;
//
//    std::cout << test << std::endl;



    clock_t end = clock();
    cout << (double)(end-start)/CLOCKS_PER_SEC << " seconds have been spent." << endl;

}