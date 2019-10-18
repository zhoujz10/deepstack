//
// Created by zhou on 19-9-24.
//

#include "turn_generator.h"


void generate_turn_data(RangeGenerator &rg, const char *s) {
    float player_range_array[hand_count];
    float oppoenet_range_array[hand_count];
    float player_values_array[hand_count];
    float opponent_values_array[hand_count];
    Board board;
    rg.generate_board(3, board);
    rg.set_board(board);
    torch::Tensor player_range = torch::zeros({hand_count}, torch::kFloat32).to(device);
    torch::Tensor opponent_range = torch::zeros({hand_count}, torch::kFloat32).to(device);
    rg.generate(player_range);
    rg.generate(opponent_range);
    int pot_size = rg.generate_pot();
//    while (pot_size < 1000)
//        pot_size = rg.generate_pot();
    int bets[2] = { pot_size, pot_size };
    Node build_tree_node( board.cards, bets );
    Resolving resolving;
    resolving.resolve_first_node(build_tree_node, player_range, opponent_range);

    memcpy(player_range_array, player_range.to(torch::kCPU).data<float>(), hand_count * sizeof(float));
    memcpy(oppoenet_range_array, opponent_range.to(torch::kCPU).data<float>(), hand_count * sizeof(float));
    memcpy(player_values_array, resolving.get_root_cfv_both_players()[0].to(torch::kCPU).data<float>(), hand_count * sizeof(float));
    memcpy(opponent_values_array, resolving.get_root_cfv_both_players()[1].to(torch::kCPU).data<float>(), hand_count * sizeof(float));

    ofstream f_write(s, ios::binary | ios::app);
    f_write.write( reinterpret_cast<char *>(&bets[0]), sizeof(int) );
    f_write.write( reinterpret_cast<char *>(&board.cards[0]), 4 * sizeof(int) );
    f_write.write( reinterpret_cast<char *>(&player_range_array[0]), hand_count * sizeof(float) );
    f_write.write( reinterpret_cast<char *>(&oppoenet_range_array[0]), hand_count * sizeof(float) );
    f_write.write( reinterpret_cast<char *>(&player_values_array[0]), hand_count * sizeof(float) );
    f_write.write( reinterpret_cast<char *>(&opponent_values_array[0]), hand_count * sizeof(float) );
    f_write.close();
}
