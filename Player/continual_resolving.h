//
// Created by zhou on 19-6-13.
//

#ifndef DEEPSTACK_CPP_CONTINUAL_RESOLVING_H
#define DEEPSTACK_CPP_CONTINUAL_RESOLVING_H


#include <boost/property_tree/ptree.hpp>
#include "../Lookahead/resolving.h"


using namespace boost::property_tree;

class continual_resolving {

public:
    Resolving *resolving = nullptr;
    Resolving *first_node_resolving = nullptr;
    torch::Tensor starting_player_range;
    torch::Tensor starting_cfvs_p2;

    void resolve_first_node();
    void start_new_hand(ptree& state);
    void compute_action(ptree& node, ptree& state);

private:
    void _resolve_node(ptree& node, ptree& state);
    void _update_invariant(ptree& node, ptree& state);
    void _sample_bet(ptree& node, ptree& state);

};


#endif //DEEPSTACK_CPP_CONTINUAL_RESOLVING_H
