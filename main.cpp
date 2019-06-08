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
#include "Lookahead/resolving.h"



int main() {

    auto card_tools = get_card_tools();

//    clock_t start = clock();


    auto n = chrono::system_clock::now();
    auto m = n.time_since_epoch();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(m).count();
    auto msecs = diff % 1000;

    std::time_t t = std::chrono::system_clock::to_time_t(n);
    cout << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S") << "." << msecs << endl;


//    auto device = c10::Device(c10::DeviceType::CUDA);


//    float a[][3] = {{4,3,2},{1,5,6}};
//    torch::Tensor tensor = torch::from_blob(a, {2, 3}, torch::kFloat32);
//    std::cout << tensor.sizes() << endl;
//    tensor = tensor.slice(1, 0, 2);
//    std::cout << tensor << std::endl;
//
//    torch::Tensor tensor_2 = torch::zeros({hand_count, hand_count}, torch::kFloat32);
//    CardTools& c = get_card_tools();
//    auto te = get_terminal_equity();

//    torch::Tensor t2;
//
//    t2 = tensor.view({1, 6});
//
//    t2[0] = 100;
//
//    std::cout << t2 << endl;
//    std::cout << tensor << endl;
//
//
//    std::cout << tensor.sum(0) << std::endl;
//    std::cout << tensor.sum(1) << std::endl;
//
//

//    PokerTreeBuilder ptb;
//    auto root_node = ptb.build_tree(build_tree_params);
//    for (auto& child : *root_node->children)
//        std::cout << child.bets[0] << ' ' << child.bets[1] << std::endl;
//    std::cout << (*(*root_node->children)[1].children)[1].children->size() << std::endl;
//
//    std::cout << ceil(10) << std::endl;
//    std::cout << ceil(10.12) << std::endl;

//    tensor[tensor > 0].fill_(1);
//    std::cout << tensor << std::endl;

//    tensor = tensor.toType(torch::kByte);
//    tensor.masked_fill_(tensor>2, 0);
//    std::cout << tensor << std::endl;

//
//    tensor.fill_(1);
//    std::cout << tensor << std::endl;


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
//    test[4][1] = 1;
//    test[3] = 2;
//    print(test[-1]);
//
//    test.slice(0, 3, -1, 1) = 3;
//
//    std::cout << test << std::endl;



    int cards[5] = {  5, 45, 11, 43, -1 };
    int bets[2] = { 100, 100 };
    Node build_tree_node( cards, bets );
    build_tree_node.current_player = constants.players.P1;
    torch::Tensor player_range = torch::zeros(hand_count, torch::kFloat32).to(device);
    torch::Tensor opponent_range = torch::zeros(hand_count, torch::kFloat32).to(device);

    card_tools.get_uniform_range(build_tree_node.board, player_range);
    card_tools.get_uniform_range(build_tree_node.board, opponent_range);

    Resolving resolving;
    std::map<std::string, torch::Tensor> results;
    resolving.resolve_first_node(build_tree_node, player_range, opponent_range, results);

    std::cout << results["root_cfvs_both_players"] << std::endl;



//    float a[6] = {1,2,3,4,5,6};
//    torch::Tensor tensor = torch::from_blob(a, {2,3}, torch::kFloat32);
//    tensor.masked_fill_(tensor.eq(1), 100);
//    std::cout << tensor << endl;


//    clock_t end = clock();
//    cout << (double)(end-start)/CLOCKS_PER_SEC << " seconds have been spent." << endl;

    n = chrono::system_clock::now();
    m = n.time_since_epoch();
    diff = std::chrono::duration_cast<std::chrono::milliseconds>(m).count();
    msecs = diff % 1000;

    t = std::chrono::system_clock::to_time_t(n);
    cout << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S") << "." << msecs << endl;

}