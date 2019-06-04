#ifndef DEEPSTACK_CPP_GAME_SETTINGS_H
#define DEEPSTACK_CPP_GAME_SETTINGS_H


const bool gpu = false;

const int streets_count = 4;

const int ante = 100;

const int stack = 20000;

const int cfr_iters[5] = { -1, 1000, 1000, 1000, 2000 };

const int cfr_skip_iters[5] = { -1, 980, 500, 500, 1000 };

const int suit_count = 4;

const int rank_count = 13;

const int card_count = suit_count * rank_count;

const int hand_count = (int)(0.5 * card_count * (card_count - 1));

const int board_card_count[5] = { -1, 0, 3, 4, 5 };

const int player_count = 2;

struct Players {
    int chance;
    int P1;
    int P2;
    Players() {
        chance = 2;
        P1 = 0;
        P2 = 1;
    }
};

struct NodeTypes {
    int terminal_fold;
    int terminal_call;
    int check;
    int chance_node;
    int inner_node;
    NodeTypes() {
        terminal_fold = -2;
        terminal_call = -1;
        check = -1;
        chance_node = 0;
        inner_node = 1;
    }
};

struct Actions {
    int fold;
    int ccall;
    Actions() {
        fold = -2;
        ccall = -1;
    }
};

struct Constants {
    int players_count;
    int streets_counts;
    Players players;
    NodeTypes node_types;
    Actions actions;
    Constants() {
        players_count = 2;
        streets_counts = 4;
    }
};

const struct Constants constants;


#endif //DEEPSTACK_CPP_GAME_SETTINGS_H
