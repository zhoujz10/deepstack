//
// Created by zhou on 19-5-30.
//

#ifndef DEEPSTACK_CPP_RUNTIME_H
#define DEEPSTACK_CPP_RUNTIME_H

//
// Created by zhou on 18-4-26.
//


#include <iostream>
#include <random>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <map>
#include <cmath>
#include <array>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "hash/SevenEval.h"
#include "io.h"
#include "runtime_constants.h"


void quicksort(int length, int a[], int sorted_hand_value[]);

template <class T>
float computeEMD(int k_next_turn, const T* histograms,
                 const float* meanNormalized, const float* cent_distance);

void sort_hand_values(int hand_value[1326], int sorted_hand_value[1326], int rank[1326]);

int get_table_id_in_3(vector<int> &state);

int get_table_id_in_4(vector<int> &state);

int get_table_id_in_5(vector<int> &state);

int get_board_idx_cpp(int table_0, int table_1, int table_2, int table_3, int table_4);

void read_values(uint8_t table_1, uint8_t table_2, uint8_t table_3, uint32_t* values);

void evaluate_hand_street_2_cpp(int table[5], int hand_values[1176][1326]);

void evaluate_hand_street_3_cpp(int table[5], int hand_values[48][1326]);

void evaluate_hand_street_4_cpp(int table[5], int hand_values[1326]);

void evaluate_hand_cpp(int table[5], int hand_values[1326]);

void set_call_matrix_2_cpp(int table[5], float equity_matrix[][1326]);

void set_call_matrix_3_cpp(int table[5], float equity_matrix[][1326]);

void set_call_matrix_4_cpp(int table[5], float equity_matrix[][1326]);

void set_call_matrix_cpp(int table[5], float equity_matrix[][1326]);

void set_call_matrix_street_3_cpp(
        int table[5],
        float equity_matrix[][1326],
        float equity_matrix_next_street[][1326][1326],
        int sorted_hand_values[][1326],
        int ranks[][1326]);

void set_fold_matrix_next_street_cpp(int board[4], float fold_matrix_next_street[][1326][1326]);

void set_fold_board_next_street_cpp(int board[4], int fold_board_next_street[][1326]);

void evalFold(
        const float oppProbs[1326],
        const int fold_board_next_street[1326],
        float retVal[1326] );

void fold_value_next_street_cpp(int num, float ranges[][48][2][1326], int fold_board_next_street[][1326], float result[][48][2][1326]);

void evalCall(
        const int hands[1326],
        const int rank[1326],
        const float oppProbs[1326],
        float retVal[1326] );

void call_value_next_street_cpp(int num, float ranges[][48][2][1326], int sorted_hand_values[][1326], int ranks[][1326], float result[][48][2][1326]);

int get_river_bucket_cpp(int board[5], float river_bucket[1326][136]);

int set_river_abstract_cpp(int board[4], float river_hand_abstract[][1326][149],
        float equity_matrix_next_street[][149][149], float fold_matrix_next_street[][149][149]);

int set_river_abstract_combine_cpp(int board[4], float river_hand_abstract[][1326][149],
        float equity_matrix_next_street[][149], float fold_matrix_next_street[][149]);

uint8_t comp_state(vector<uint8_t> &state, vector<uint8_t> &tmp);

void get_min_state_5(vector<uint8_t> &state);

int get_state_id_in_3(int table_hand[5]);

int get_state_id_in_4(int table_hand[6]);

bool floatLEQ(float a, float b);

void get_river_win_tie_cpp(int table[5], int hand_values[1326], int hand_win_tie[][1326]);

void load_hand2bucket_single_turn_cpp(int table[4], unsigned short board_abs_counts[48], short hand2abstract[][1326],
        unsigned short abs2hand[][1081], unsigned short abs_pos[][149], unsigned short abs_count[][149][2],
        unsigned short hand_collide_count[][1326][3]);

int toHash (const char* pString);

#endif //DEEPSTACK_CPP_RUNTIME_H
