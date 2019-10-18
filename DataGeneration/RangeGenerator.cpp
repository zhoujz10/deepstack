//
// Created by zhou on 19-9-24.
//

#include "RangeGenerator.h"


bool cmp_second_asc(pair<int, int> a, pair<int, int> b) {
    return a.second < b.second;
}

void RangeGenerator::_generate_recursion(float* sorted_ranges, const float mass, const int start, const int end) {
    int _card_count = end - start;
    if (_card_count == 1)
        sorted_ranges[start] = mass;
    else {
        float r = dice();
        auto mass1 = mass * r;
        auto mass2 = mass - mass1;
        int half_size = _card_count / 2;
        if (_card_count % 2 != 0)
            half_size += (int)round(dice());
        _generate_recursion(sorted_ranges, mass1, start, start + half_size);
        _generate_recursion(sorted_ranges, mass2, start + half_size, end);
    }
}

void RangeGenerator::_generate_sorted_range(float* sorted_ranges) {
    _generate_recursion(sorted_ranges, 1, 0, possible_hands_count);
}

void RangeGenerator::set_board(const Board& board) {
    int table[5];
    memcpy(table, board.cards, 5 * sizeof(int));
    int hand_values[hand_count];
    evaluate_hand_cpp(table, hand_values);
    for (int i=0; i<hand_count; ++i)
        hand_values[i] = 1 - hand_values[i];

    CardTools::get_possible_hand_indexes(board, possible_hands_mask);
    possible_hands_count = 0;
    for (int i=0; i<hand_count; ++i)
        possible_hands_count += possible_hands_mask[i];

    std::vector<pair<int, int>>().swap(hand_strengths);
    assert(hand_strengths.size() == 0);
    for (int i=0; i<hand_count; ++i) {
        hand_strengths.push_back(pair<int, int>(i, hand_values[i]));
    }
    sort(hand_strengths.begin(), hand_strengths.end(), cmp_second_asc);
}

void RangeGenerator::generate(torch::Tensor& ranges) {
    auto sorted_ranges = new float[possible_hands_count];
    _generate_sorted_range(sorted_ranges);
    float reordered_ranges[hand_count];
    memset(reordered_ranges, 0, hand_count * sizeof(float));
    for (int i=0; i<hand_count; ++i) {
        if (hand_strengths[i].second == 1)
            break;
        reordered_ranges[hand_strengths[i].first] = sorted_ranges[i];
    }
    ranges.copy_(torch::from_blob(reordered_ranges, {hand_count}, torch::kFloat32).to(device));
//    std::cout << ranges.sum() << std::endl;
//    assert(ranges.sum().to(torch::kCPU).data<float>()[0] > 0.999 && ranges.to(torch::kCPU).sum().data<float>()[0] < 1.001);
}

void RangeGenerator::generate_board(const int street, Board& board) {
    std::vector<int> iv(card_count, 0);
    for (int i=0; i<card_count; ++i)
        iv[i] = i;
    std::shuffle(iv.begin(), iv.end(), std::default_random_engine(clock()));
    for (int i=0; i<board_card_count[street]; ++i)
        board.cards[i] = iv[i];
}

int RangeGenerator::generate_pot() {
    float r = dice();
    int pot_size_idx = (int)(r * 5);
    int mul_idx = (int)((r - 0.2 * pot_size_idx) * random_potsize_numbers[pot_size_idx]);
    int pot_size = random_potsize_starts[pot_size_idx] + 50 * mul_idx;
    assert(pot_size >= 100 && pot_size < 20000);
    return pot_size;
}



