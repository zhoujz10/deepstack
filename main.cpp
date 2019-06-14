#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <cmath>
#include <time.h>

#include "Web/server_http.hpp"
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <vector>


#include "TerminalEquity/terminal_equity.h"
#include "Game/board.h"
#include "Tree/node.h"
#include "Game/card_tools.h"
#include "Game/bet_sizing.h"
#include "Tree/poker_tree_builder.h"
#include "Lookahead/resolving.h"


using namespace boost::property_tree;
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;


int main() {

    auto card_tools = get_card_tools();
    get_turn_value();

    HttpServer server;
    server.config.port = 8080;

    server.resource["^/json$"]["POST"] = [](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
        try {
            ptree pt;
            read_json(request->content, pt);

            std::cout << pt.get<int>("pot") << std::endl;


            auto name = pt.get<string>("firstName") + " " + pt.get<string>("lastName");

            *response << "HTTP/1.1 200 OK\r\n"
                      << "Content-Length: " << name.length() << "\r\n\r\n"
                      << name;
        }
        catch (const exception &e) {
            *response << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << strlen(e.what()) << "\r\n\r\n"
                      << e.what();
        }
    };

    std::cout << "Server ready at port " << server.config.port << "." << std::endl;

    thread server_thread([&server]() {
        // Start server
        server.start();
    });
    server_thread.join();




//    int cards[5];
//    int bets[2];
//
//    for (int i=0; i<5; ++i)
//        cin >> cards[i];
//
//    for (int i=0; i<2; ++i)
//        cin >> bets[i];


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








//    int cards[5] = {  -1, -1, -1, -1, -1 };
////    int cards[5] = {  5, 45, 11, -1, -1 };
////    int cards[5] = {  5, 45, 11, 43, -1 };
////    int cards[5] = {  5, 15, 25, 35, -1 };
//    int bets[2] = { 1000, 1000 };
//    Node build_tree_node( cards, bets );
//    build_tree_node.current_player = constants.players.P1;
//    torch::Tensor player_range = torch::zeros(hand_count, torch::kFloat32).to(device);
//    torch::Tensor opponent_range = torch::zeros(hand_count, torch::kFloat32).to(device);
//
//    card_tools.get_uniform_range(build_tree_node.board, player_range);
//    card_tools.get_uniform_range(build_tree_node.board, opponent_range);
//
//    Resolving resolving;
//    resolving.resolve_first_node(build_tree_node, player_range, opponent_range);
//
//    std::cout << resolving.get_root_cfv_both_players() << std::endl;







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