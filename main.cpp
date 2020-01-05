#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include <random>

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
#include "Game/card_to_string_conversion.h"
#include "Game/bet_sizing.h"
#include "Tree/poker_tree_builder.h"
#include "Lookahead/resolving.h"
#include "Player/continual_resolving.h"
#include "DataGeneration/turn_generator.h"


using namespace boost::property_tree;
using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;


class InputParser{
public:
    InputParser (int &argc, char **argv) {
        for (int i=1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }

    const std::string& getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()){
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    bool cmdOptionExists(const std::string &option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option)
               != this->tokens.end();
    }
private:
    std::vector <std::string> tokens;
};

int main(int argc, char* argv[]) {

//    NextRoundValue& flop_value = get_flop_value();
//    print(flop_value.board_buckets.slice(0, 100, 101, 1));
//    return 0;

//    torch::Tensor a = torch::ones({1, 2001}, torch::kFloat32).to(device) / 1000;
//    torch::Tensor b = torch::zeros({1, 2000}, torch::kFloat32).to(device);
//    auto flop_net = &get_flop_nn();
//    flop_net->get_value(a, b);
//    auto turn_net = &get_turn_nn();
//    turn_net->get_value(a, b);


//    torch::Tensor a = torch::ones({1, 169*2+1}, torch::kFloat32).to(device) / 169;
//    torch::Tensor b = torch::zeros({1, 169*2}, torch::kFloat32).to(device);
//    auto aux_net = &get_aux_nn();
//    aux_net->get_value(a, b);
//    print(b);
//    return 0;


//    generate_mode = true;
//    RangeGenerator rg;
//    char s[100];
//    std::function<float()> _dice = std::bind(distribution, generator);
//    auto filename_rand = (int)(10000000 * _dice());
//    sprintf(s, "output_turn_cpp_700.bin.%d", filename_rand);
//
//    for (int i=0; i<100000; ++i) {
//        if (i % 100 == 0)
//            std::cout << i << std::endl;
//        generate_turn_data(rg, s);
//    }
//
//    return 0;

    HttpServer server;
    server.config.port = 8080;

    InputParser input(argc, argv);

    if(input.cmdOptionExists("--port")){
        const std::string &port = input.getCmdOption("--port");
        server.config.port = std::stoi(port);
    }

    if(input.cmdOptionExists("--cache")){
        const std::string &use_cache = input.getCmdOption("--cache");
        params::use_cache = std::stoi(use_cache);
    }

    if(input.cmdOptionExists("--ante")){
        const std::string &_ante = input.getCmdOption("--ante");
        minimum_ante = std::stoi(_ante);
        ante = minimum_ante;
    }

    if(input.cmdOptionExists("--add")){
        const std::string &additional_ante = input.getCmdOption("--add");
        params::minimum_additional_ante = std::stoi(additional_ante);
        params::additional_ante = params::minimum_additional_ante;
    }

    pokermaster = input.cmdOptionExists("--pokermaster");


//    used to debug
//    params::use_cache = 0;
//    minimum_ante = 100;
//    ante = 100;
//    params::minimum_additional_ante = 0;
//    params::additional_ante = 0;
//    params::stack = 20000;
//    pokermaster = false;


    if (pokermaster) {
        if (params::additional_ante) {
            preflop_cache_root_file = preflop_cache_root_file_pokermaster_addante;
            preflop_cache_root_file += "_";
            preflop_cache_root_file += std::to_string(minimum_ante);
            preflop_cache_root_file += "_";
            preflop_cache_root_file += std::to_string(params::minimum_additional_ante);
            preflop_cache_root_file += "/";
        }
        else
            preflop_cache_root_file = preflop_cache_root_file_pokermaster;
    }
    std::cout << preflop_cache_root_file << std::endl;

    CardToString& card_to_string = get_card_to_string();

    auto card_tools = get_card_tools();
    get_flop_value();
    get_turn_value();

    ContinualResolving continual_resolving;

    server.resource["^/get_action"]["POST"] = [&card_to_string, &continual_resolving](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
        try {

            auto n = chrono::system_clock::now();
            auto m = n.time_since_epoch();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(m).count();
            auto msecs = diff % 1000;

            std::time_t t = std::chrono::system_clock::to_time_t(n);
            auto tm_start = std::localtime(&t);
            auto start_time = (float)tm_start->tm_sec + (float)msecs / 1000;

            ptree pt;
            Node node;
            read_json(request->content, pt);
            pt.put<int>("hand_id", card_to_string.string_to_hand(pt.get<string>("hand")));

            if (pt.get<int>("new_game")) {
//                reset game settings
                ante = minimum_ante;
                params::additional_ante = params::minimum_additional_ante;
                params::position = pt.get<int>("position");
                continual_resolving.start_new_hand(pt);
            }

            rate = pt.get<int>("ante") / ante;
            pt.add("rate", rate);
            std::cout << "pt_rate: " << pt.get<int>("rate") << std::endl;
            params::stack = pt.get<int>("stack") / rate;

            node.terminal = false;
            node.current_player = pt.get<int>("position");
            node.street = pt.get<int>("street");
            card_to_string.string_to_board(pt.get<string>("board"), node.board);

            int bets[2];

            if (pt.get<int>("bet_0") % rate == 0 && pt.get<int>("bet_1") % rate == 0) {
                bets[0] = min(pt.get<int>("bet_0") / rate, params::stack);
                bets[1] = min(pt.get<int>("bet_1") / rate, params::stack);
                pt.add("need_rate_resume", false);
            }
            else {
                rate = 1;
//                params::additional_ante *= rate;
//                ante = pt.get<int>("ante");
//                params::stack = pt.get<int>("stack");
                bets[0] = min(pt.get<int>("bet_0"), pt.get<int>("stack"));
                bets[1] = min(pt.get<int>("bet_1"), pt.get<int>("stack"));
                pt.add("need_rate_resume", true);
            }

            node.set_bets(bets);

            int adviced_action = continual_resolving.compute_action(node, pt);

            adviced_action = adviced_action >= 0 ? adviced_action * rate : adviced_action;

            if ((pt.get<int>("street") == 1 && abs(adviced_action - pt.get<int>("stack")) <= 2 * rate) ||
            abs(adviced_action - pt.get<int>("stack")) <= rate)
                adviced_action = pt.get<int>("stack");

            auto action = std::to_string(adviced_action);

            n = chrono::system_clock::now();
            m = n.time_since_epoch();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(m).count();
            msecs = diff % 1000;

            t = std::chrono::system_clock::to_time_t(n);
            auto tm_end = std::localtime(&t);
            auto end_time = (float)tm_end->tm_sec + (float)msecs / 1000;
            auto computation_time = end_time - start_time;
            if (computation_time < 0)
                computation_time += 60;
            std::cout << "Used time: " << computation_time << " seconds." << std::endl;

            *response << "HTTP/1.1 200 OK\r\n"
                      << "Content-Length: " << action.length() << "\r\n\r\n"
                      << action;
        }
        catch (const exception &e) {
            *response << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << strlen(e.what()) << "\r\n\r\n"
                      << e.what();
        }
    };

    std::cout << "Server ready at port " << server.config.port << "." << std::endl;
    if (params::use_cache == 1)
        std::cout << "The algorithm uses preflop cache." << std::endl;
    else
        std::cout << "The algorithm does not use preflop cache." << std::endl;
    std::cout << "The additional ante of the game is " << (float)params::additional_ante / ante << "bb." << std::endl;
    if (pokermaster)
        std::cout << "The algorithm is used for pokermaster." << std::endl;
    else
        std::cout << "The algorithm is used for slumbot." << std::endl;

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

//    auto n = chrono::system_clock::now();
//    auto m = n.time_since_epoch();
//    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(m).count();
//    auto msecs = diff % 1000;
//
//    std::time_t t = std::chrono::system_clock::to_time_t(n);
//    cout << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S") << "." << msecs << endl;


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





////    int cards[5] = {  -1, -1, -1, -1, -1 };
//    int cards[5] = {  5, 45, 11, -1, -1 };
////    int cards[5] = {  5, 45, 11, 43, -1 };
////    int cards[5] = {  5, 15, 25, 35, -1 };
//    int bets[2] = { 100, 100 };
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

//    n = chrono::system_clock::now();
//    m = n.time_since_epoch();
//    diff = std::chrono::duration_cast<std::chrono::milliseconds>(m).count();
//    msecs = diff % 1000;
//
//    t = std::chrono::system_clock::to_time_t(n);
//    cout << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S") << "." << msecs << endl;

}