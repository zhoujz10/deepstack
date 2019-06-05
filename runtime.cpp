//
// Created by zhou on 19-6-5.
//

#include "runtime.h"


void quicksort(int length, int a[], int sorted_hand_value[]) {
    int i, j, temp;

    for(i = 0; i < length; ++i){
        for(j = length - 1; j > i; --j){
            if(a[j] < a[j - 1]){
                temp = a[j];
                a[j] = a[j - 1];
                a[j - 1] = temp;

                temp = sorted_hand_value[j];
                sorted_hand_value[j] = sorted_hand_value[j - 1];
                sorted_hand_value[j - 1] = temp;
            }
        }
    }
}

template <class T>
float computeEMD(int k_next_turn, const T* histograms,
                 const float* meanNormalized, const float* cent_distance) {
    float cum=0, totCost=0;
    for (int i=0; i<k_next_turn-1; ++i) {
        cum += meanNormalized[i] - histograms[i];
        totCost += abs(cum) * cent_distance[i];
    }
    return totCost;
}

void sort_hand_values(int hand_value[1326], int sorted_hand_value[1326], int rank[1326]) {
    memcpy(sorted_hand_value, hand_idx, 1326*sizeof(int));
    memcpy(rank, hand_value, 1326*sizeof(int));
    quicksort(1326, rank, sorted_hand_value);
}

int get_table_id_in_3(vector<int> &state) {
    sort(state.begin(), state.end());

    // get_table_id
    int table_id = 0;
    table_id += table_3_num[state[0]];
    table_id += table_4_num[state[1]] - table_4_num[state[0]+1];
    table_id += state[2] - state[1] - 1;

    return table_id;
}

int get_table_id_in_4(vector<int> &state) {

    // get_table_id
    int table_id = -1;

    for (int table_3=0; table_3<52; ++table_3) {
        if (table_3 == state[0] || table_3 == state[1] || table_3 == state[2])
            continue;
        table_id ++;
        if (table_3 == state[3])
            return table_id;
    }
}

int get_table_id_in_5(vector<int> &state) {

    // get_table_id
    int table_id = -1;

    for (int table_4=0; table_4<52; ++table_4) {
        if (table_4 == state[0] || table_4 == state[1] || table_4 == state[2] || table_4 == state[3])
            continue;
        table_id ++;
        if (table_4 == state[4])
            return table_id;
    }
}

int get_board_idx_cpp(int table_0, int table_1, int table_2, int table_3, int table_4) {
    if (table_3 == -1) {
        vector<int> table = {table_0, table_1, table_2};
        return get_table_id_in_3(table);
    }
    else if (table_4 == -1) {
        vector<int> table = {table_0, table_1, table_2, table_3};
        return get_table_id_in_4(table);
    }
    else {
        vector<int> table = {table_0, table_1, table_2, table_3, table_4};
        return get_table_id_in_5(table);
    }

    return 0;
}

void read_values(uint8_t table_1, uint8_t table_2, uint8_t table_3, uint32_t* values) {
    string folder = "cython_lib/flop_values";
    string slash = "/";
    string underline = "_";
    string str = folder + slash + to_string(table_1) + slash + to_string(table_1) + underline +
                 to_string(table_2) + underline + to_string(table_3) + ".bin";
    ifstream f_read(str.c_str(), ios::binary);
    unsigned size;
    f_read.read( reinterpret_cast<char *>(&size), sizeof(unsigned) );
    f_read.read( reinterpret_cast<char *>(&values[0]), size*sizeof(uint32_t) );
    f_read.close();
}

void evaluate_hand_street_2_cpp(int table[5], int hand_values[1176][1326]) {
    for (int k=0; k<1176; ++k)
        for (int i=0; i<HAND_COUNT; ++i)
            hand_values[k][i] = 0;

    uint8_t table_0 = (uint8_t)table[0];
    uint8_t table_1 = (uint8_t)table[1];
    uint8_t table_2 = (uint8_t)table[2];
    int k = -1;
    for (uint8_t table_3=0; table_3<52; ++table_3)
        for (uint8_t table_4=table_3+1; table_4<52; ++table_4) {
            if (table_3 == table_0 || table_3 == table_1 || table_3 == table_2 ||
                table_4 == table_0 || table_4 == table_1 || table_4 == table_2)
                continue;
            k ++;
            for (int i=0; i<HAND_COUNT; ++i) {
                uint8_t hand_1 = first_card[i];
                uint8_t hand_2 = second_card[i];
                if (hand_1 == table_0 || hand_1 == table_1 || hand_1 == table_2 || hand_1 == table_3 || hand_1 == table_4)
                    continue;
                if (hand_2 == table_0 || hand_2 == table_1 || hand_2 == table_2 || hand_2 == table_3 || hand_2 == table_4)
                    continue;
                hand_values[k][i] = SevenEval::GetRank(table_0, table_1, table_2, table_3, table_4, hand_1, hand_2);
            }
        }
}

void evaluate_hand_street_3_cpp(int table[5], int hand_values[48][1326]) {
    for (int k=0; k<48; ++k)
        for (int i=0; i<HAND_COUNT; ++i)
            hand_values[k][i] = 0;

    uint8_t table_0 = (uint8_t)table[0];
    uint8_t table_1 = (uint8_t)table[1];
    uint8_t table_2 = (uint8_t)table[2];
    uint8_t table_3 = (uint8_t)table[3];
    int k = -1;
    for (uint8_t table_4=0; table_4<52; ++table_4) {
        if (table_4 == table_0 || table_4 == table_1 || table_4 == table_2 || table_4 == table_3)
            continue;
        k ++;
        for (int i=0; i<HAND_COUNT; ++i) {
            uint8_t hand_1 = first_card[i];
            uint8_t hand_2 = second_card[i];
            if (hand_1 == table_0 || hand_1 == table_1 || hand_1 == table_2 || hand_1 == table_3 || hand_1 == table_4)
                continue;
            if (hand_2 == table_0 || hand_2 == table_1 || hand_2 == table_2 || hand_2 == table_3 || hand_2 == table_4)
                continue;
            hand_values[k][i] = SevenEval::GetRank(table_0, table_1, table_2, table_3, table_4, hand_1, hand_2);
        }
    }
}

void evaluate_hand_street_4_cpp(int table[5], int hand_values[1326]) {
    for (int i=0; i<HAND_COUNT; ++i)
        hand_values[i] = 0;

    uint8_t table_0 = (uint8_t)table[0];
    uint8_t table_1 = (uint8_t)table[1];
    uint8_t table_2 = (uint8_t)table[2];
    uint8_t table_3 = (uint8_t)table[3];
    uint8_t table_4 = (uint8_t)table[4];
    for (int i=0; i<HAND_COUNT; ++i) {
        uint8_t hand_1 = first_card[i];
        uint8_t hand_2 = second_card[i];
        if (hand_1 == table_0 || hand_1 == table_1 || hand_1 == table_2 || hand_1 == table_3 || hand_1 == table_4)
            continue;
        if (hand_2 == table_0 || hand_2 == table_1 || hand_2 == table_2 || hand_2 == table_3 || hand_2 == table_4)
            continue;
        hand_values[i] = SevenEval::GetRank(table_0, table_1, table_2, table_3, table_4, hand_1, hand_2);
    }
}

void evaluate_hand_cpp(int table[5], int hand_values[1326]) {
    if (table[0] == -1) {
        read_pointer(hand_values, "preflop_values.bin");
        return;
    }

    for (int i=0; i<1326; ++i)
        hand_values[i] = 0;

    if (table[3] == -1) {
        uint8_t table_0 = (uint8_t)table[0];
        uint8_t table_1 = (uint8_t)table[1];
        uint8_t table_2 = (uint8_t)table[2];
        for (uint8_t table_3=0; table_3<52; ++table_3) {
            if (table_3 == table_0 || table_3 == table_1 || table_3 == table_2)
                continue;
            for (uint8_t table_4=table_3+1; table_4<52; ++table_4) {
                if (table_4 == table_0 || table_4 == table_1 || table_4 == table_2)
                    continue;
                for (int i=0; i<HAND_COUNT; ++i) {
                    uint8_t hand_1 = first_card[i];
                    uint8_t hand_2 = second_card[i];
                    if (hand_1 == table_0 || hand_1 == table_1 || hand_1 == table_2 || hand_1 == table_3 || hand_1 == table_4)
                        continue;
                    if (hand_2 == table_0 || hand_2 == table_1 || hand_2 == table_2 || hand_2 == table_3 || hand_2 == table_4)
                        continue;
                    hand_values[i] += SevenEval::GetRank(table_0, table_1, table_2, table_3, table_4, hand_1, hand_2);
                }
            }
        }
    }
    else if (table[4] == -1) {
        uint8_t table_0 = (uint8_t)table[0];
        uint8_t table_1 = (uint8_t)table[1];
        uint8_t table_2 = (uint8_t)table[2];
        uint8_t table_3 = (uint8_t)table[3];
        for (uint8_t table_4=0; table_4<52; ++table_4) {
            if (table_4 == table_0 || table_4 == table_1 || table_4 == table_2 || table_4 == table_3)
                continue;
            for (int i=0; i<HAND_COUNT; ++i) {
                uint8_t hand_1 = first_card[i];
                uint8_t hand_2 = second_card[i];
                if (hand_1 == table_0 || hand_1 == table_1 || hand_1 == table_2 || hand_1 == table_3 || hand_1 == table_4)
                    continue;
                if (hand_2 == table_0 || hand_2 == table_1 || hand_2 == table_2 || hand_2 == table_3 || hand_2 == table_4)
                    continue;
                hand_values[i] += SevenEval::GetRank(table_0, table_1, table_2, table_3, table_4, hand_1, hand_2);
            }
        }
    }
    else
        evaluate_hand_street_4_cpp(table, hand_values);
}

void set_call_matrix_2_cpp(int table[5], float equity_matrix[][1326]) {
    int hand_values[1176][HAND_COUNT];
    evaluate_hand_street_2_cpp(table, hand_values);
    for (int i=0; i<HAND_COUNT; ++i) {
        equity_matrix[i][i] = 0;
        for (int j=i+1; j<HAND_COUNT; ++j) {
            for (int k=0; k<1176; ++k) {
                if (hand_values[k][i] == 0 || hand_values[k][j] == 0 || first_card[i] == first_card[j] || first_card[i] == second_card[j] || second_card[i] == first_card[j] || second_card[i] == second_card[j]) {
                    continue;
                }
                else if (hand_values[k][i] < hand_values[k][j]) {
                    equity_matrix[i][j] += 1;
                    equity_matrix[j][i] += -1;
                }
                else if (hand_values[k][i] == hand_values[k][j]) {
                    continue;
                }
                else {
                    equity_matrix[i][j] += -1;
                    equity_matrix[j][i] += 1;
                }
            }
        }
    }

    for (int i=0; i<HAND_COUNT; ++i)
        for (int j=0; j<HAND_COUNT; ++j)
            equity_matrix[i][j] /= 990;
}

void set_call_matrix_3_cpp(int table[5], float equity_matrix[][1326]) {
    int hand_values[48][HAND_COUNT];
    evaluate_hand_street_3_cpp(table, hand_values);
    for (int i=0; i<HAND_COUNT; ++i) {
        equity_matrix[i][i] = 0;
        for (int j=i+1; j<HAND_COUNT; ++j) {
            for (int k=0; k<48; ++k) {
                if (hand_values[k][i] == 0 || hand_values[k][j] == 0 || first_card[i] == first_card[j] || first_card[i] == second_card[j] || second_card[i] == first_card[j] || second_card[i] == second_card[j]) {
                    continue;
                }
                else if (hand_values[k][i] < hand_values[k][j]) {
                    equity_matrix[i][j] += 1;
                    equity_matrix[j][i] += -1;
                }
                else if (hand_values[k][i] == hand_values[k][j]) {
                    continue;
                }
                else {
                    equity_matrix[i][j] += -1;
                    equity_matrix[j][i] += 1;
                }
            }
        }
    }

    for (int i=0; i<HAND_COUNT; ++i)
        for (int j=0; j<HAND_COUNT; ++j)
            equity_matrix[i][j] /= 44;
}

void set_call_matrix_4_cpp(int table[5], float equity_matrix[][1326]) {
    int hand_values[HAND_COUNT];
    evaluate_hand_street_4_cpp(table, hand_values);
    for (int i=0; i<HAND_COUNT; ++i) {
        equity_matrix[i][i] = 0;
        for (int j=i+1; j<HAND_COUNT; ++j) {
            if (hand_values[i] == 0 || hand_values[j] == 0 || first_card[i] == first_card[j] || first_card[i] == second_card[j] || second_card[i] == first_card[j] || second_card[i] == second_card[j]) {
                continue;
            }
            else if (hand_values[i] < hand_values[j]) {
                equity_matrix[i][j] += 1;
                equity_matrix[j][i] += -1;
            }
            else if (hand_values[i] == hand_values[j]) {
                continue;
            }
            else {
                equity_matrix[i][j] += -1;
                equity_matrix[j][i] += 1;
            }
        }
    }
}

void set_call_matrix_cpp(int table[5], float equity_matrix[][1326]) {

    if (table[3] == -1)
        set_call_matrix_2_cpp(table, equity_matrix);
    else if (table[4] == -1)
        set_call_matrix_3_cpp(table, equity_matrix);
    else
        set_call_matrix_4_cpp(table, equity_matrix);
}

void set_call_matrix_street_3_cpp(
        int table[5],
        float equity_matrix[][1326],
        float equity_matrix_next_street[][1326][1326],
        int sorted_hand_values[][1326],
        int ranks[][1326])
{
    int hand_values[48][1326];
    evaluate_hand_street_3_cpp(table, hand_values);

    for (int k=0; k<48; ++k)
        for (int i=0; i<1326; ++i) {
            equity_matrix_next_street[k][i][i] = 0;
            for (int j=i+1; j<1326; ++j)
                if (hand_values[k][i] == 0 || hand_values[k][j] == 0 || first_card[i] == first_card[j] || first_card[i] == second_card[j] || second_card[i] == first_card[j] || second_card[i] == second_card[j]) {
                    equity_matrix_next_street[k][i][j] = 0;
                    equity_matrix_next_street[k][j][i] = 0;
                }
                else if (hand_values[k][i] < hand_values[k][j]) {
                    equity_matrix_next_street[k][i][j] = 1;
                    equity_matrix_next_street[k][j][i] = -1;
                }
                else if (hand_values[k][i] == hand_values[k][j]) {
                    equity_matrix_next_street[k][i][j] = 0;
                    equity_matrix_next_street[k][j][i] = 0;
                }
                else {
                    equity_matrix_next_street[k][i][j] = -1;
                    equity_matrix_next_street[k][j][i] = 1;
                }
        }

    for (int i=0; i<48; ++i)
        sort_hand_values(hand_values[i], sorted_hand_values[i], ranks[i]);
}

void set_fold_matrix_next_street_cpp(int board[4], float fold_matrix_next_street[][1326][1326]) {
    int k = -1;
    for (uint8_t board_4=0; board_4<52; ++board_4) {
        if (board_4 == board[0] || board_4 == board[1] || board_4 == board[2] || board_4 == board[3])
            continue;
        k ++;
        for (int i=0; i<HAND_COUNT; ++i) {
            uint8_t hand_1 = first_card[i];
            uint8_t hand_2 = second_card[i];
            if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4) {
                for (int j=0; j<HAND_COUNT; ++j) {
                    fold_matrix_next_street[k][i][j] = 0;
                    fold_matrix_next_street[k][j][i] = 0;
                }
            }
            for (int j=i; j<HAND_COUNT; ++j) {
                uint8_t hand_3 = first_card[j];
                uint8_t hand_4 = second_card[j];
                if (hand_1 == hand_3 || hand_1 == hand_4 || hand_2 == hand_3 || hand_2 == hand_4) {
                    fold_matrix_next_street[k][i][j] = 0;
                    fold_matrix_next_street[k][j][i] = 0;
                }
            }
        }
    }
}

void set_fold_board_next_street_cpp(int board[4], int fold_board_next_street[][1326]) {
    int k = -1;
    for (uint8_t board_4=0; board_4<52; ++board_4) {
        if (board_4 == board[0] || board_4 == board[1] || board_4 == board[2] || board_4 == board[3])
            continue;
        k ++;
        for (int i=0; i<HAND_COUNT; ++i) {
            uint8_t hand_1 = first_card[i];
            uint8_t hand_2 = second_card[i];
            if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4)
                fold_board_next_street[k][i] = 0;
        }
    }
}

void evalFold(
        const float oppProbs[1326],
        const int fold_board_next_street[1326],
        float retVal[1326] )
{
    float sum;
    int hand;
    float sumIncludingCard[ 52 ];

    memset( sumIncludingCard, 0, sizeof( sumIncludingCard ) );
    sum = 0;

    /* One pass over the opponent's hands to build up sums
    * and probabilities for the inclusion / exclusion evaluation */
    for( hand = 0; hand < 1326; hand++ ) {

        if( oppProbs[ hand ] > 0.0 && fold_board_next_street[hand] == 1 ) {

            sum += oppProbs[ hand ];
            sumIncludingCard[ first_card[ hand ] ] += oppProbs[ hand ];
            sumIncludingCard[ second_card[ hand ] ] += oppProbs[ hand ];
        }
    }

    /* One pass over our hands to assign values */
    for( hand = 0; hand < 1326; hand++ ) {

        if( fold_board_next_street[hand] == 1 ) {

            retVal[ hand ] = sum
                             - sumIncludingCard[ first_card[ hand ] ]
                             - sumIncludingCard[ second_card[ hand ] ]
                             + oppProbs[ hand ];

        }
    }
}

void fold_value_next_street_cpp(int num, float ranges[][48][2][1326], int fold_board_next_street[][1326], float result[][48][2][1326]) {

    for (int i=0; i<num; ++i)
        for (int j=0; j<48; ++j)
            for (int player=0; player<2; ++player)
                evalFold(ranges[i][j][player], fold_board_next_street[j], result[i][j][player]);
}

void evalCall(
        const int hands[1326],
        const int rank[1326],
        const float oppProbs[1326],
        float retVal[1326] )
{
    /* Showdown! */
    float sum;
    int i, j, k;
    // float sumIncludingCard[ 52 ];
    float sumIncludingCard[ 52 ];

    /* Set up variables */
    sum = 0;
    memset( sumIncludingCard, 0, sizeof( sumIncludingCard ) );
    memset( retVal, 0, 1326*sizeof( float ) );

    /* Consider us losing to everything initially */
    for( k = 0; k < 1326; k++ ) {

        if( oppProbs[ hands[k] ] > 0.0 && rank[k] != 0 ) {

            sumIncludingCard[ first_card[ hands[k] ] ] -= oppProbs[ hands[k] ];
            sumIncludingCard[ second_card[ hands[k] ] ] -= oppProbs[ hands[k] ];
            sum -= oppProbs[ hands[k] ];
        }
    }

    for( i = 0; i < 1326; ) {

        /* hand i is first in a group of ties; find the last hand in the group */
        for( j = i + 1;
             ( j < 1326 ) && ( rank[ j ] == rank[ i ] );
             j++ );

        /* Move all tied hands from the lose group to the tie group */
        for( k = i; k < j; k++ ) {

            if (rank[k] == 0)
                continue;

            sumIncludingCard[ first_card[ hands[k] ] ] += oppProbs[ hands[k] ];
            sumIncludingCard[ second_card[ hands[k] ] ] += oppProbs[ hands[k] ];
            sum += oppProbs[ hands[k] ];
        }

        /* Evaluate all hands in the tie group */
        if ( rank[i] != 0 ) {
            for( k = i; k < j; ++k ) {
                retVal[ hands[k] ] = sum
                                     - sumIncludingCard[ first_card[ hands[k] ] ]
                                     - sumIncludingCard[ second_card[ hands[k] ] ];
            }
        }

        /* Move this tie group to wins, then move to next tie group */
        for( k = i; k < j; k++ ) {

            if (rank[k] == 0)
                continue;

            sumIncludingCard[ first_card[ hands[k] ] ] += oppProbs[ hands[k] ];
            sumIncludingCard[ second_card[ hands[k] ] ] += oppProbs[ hands[k] ];
            sum += oppProbs[ hands[k] ];
        }
        i = j;
    }
}

void call_value_next_street_cpp(int num, float ranges[][48][2][1326], int sorted_hand_values[][1326], int ranks[][1326], float result[][48][2][1326]) {
    for (int i=0; i<num; ++i)
        for (int j=0; j<48; ++j)
            for (int player=0; player<2; ++player)
                evalCall(sorted_hand_values[j], ranks[j], ranges[i][j][player], result[i][j][player]);
}

int get_river_bucket_cpp(int board[5], float river_bucket[1326][136]) {
    int k = -1;
    int r, h, c, tmp;
    int ranks[136], nums[136];
    int rank2abstract[7463], hand_rank[HAND_COUNT];
    int hand_1, hand_2;
    int cur, abs_count = -1;

    k ++;
    cur = -1;
    memset( rank2abstract, -1, sizeof( rank2abstract ) );
    memset( nums, 0, sizeof( nums ) );
    memset( hand_rank, 0, sizeof( hand_rank ) );
    for (h=0; h<HAND_COUNT; ++h) {
        hand_1 = first_card[h];
        hand_2 = second_card[h];
        if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board[4] ||
            hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board[4])
            continue;
        r = SevenEval::GetRank(board[0], board[1], board[2], board[3], board[4], hand_1, hand_2);
        hand_rank[h] = r;
        if (rank2abstract[r] == -1) {
            rank2abstract[r] = ++cur;
            ranks[cur] = r;
            nums[cur] += 1;
        }
        else {
            nums[rank2abstract[r]] += 1;
        }
    }
    abs_count = cur + 1;
    for (int i=0; i<abs_count; ++i) {
        for (int j=i+1; j<abs_count; ++j) {
            if (ranks[i] > ranks[j])
                continue;
            tmp = ranks[i];
            ranks[i] = ranks[j];
            ranks[j] = tmp;
            rank2abstract[ranks[i]] = i;
            rank2abstract[ranks[j]] = j;
            tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }
    }
    for (h=0; h<HAND_COUNT; ++h) {
        r = hand_rank[h];
        if (r == 0)
            continue;
        river_bucket[h][rank2abstract[r]] = 1;
    }

    return abs_count;
}

int set_river_abstract_cpp(int board[4], float river_hand_abstract[][1326][149], float equity_matrix_next_street[][149][149], float fold_matrix_next_street[][149][149]) {
    int k = -1;
    int r, h, c;
    int ranks[149], nums[149];
    int rank2abstract[7463], hand2abstract[HAND_COUNT];
    int hand_1, hand_2;
    int cur, abs_count = -1;
    for (uint8_t board_4=0; board_4<52; ++board_4) {
        if (board_4 == board[0] || board_4 == board[1] || board_4 == board[2] || board_4 == board[3])
            continue;
        k ++;
        cur = -1;
        memset( rank2abstract, -1, sizeof( rank2abstract ) );
        memset( nums, 0, sizeof( nums ) );
        memset( hand2abstract, 0, sizeof( hand2abstract ) );
        for (h=0; h<HAND_COUNT; ++h) {
            hand_1 = first_card[h];
            hand_2 = second_card[h];
            if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4)
                continue;
            r = SevenEval::GetRank(board[0], board[1], board[2], board[3], board_4, hand_1, hand_2);
            if (rank2abstract[r] == -1) {
                rank2abstract[r] = ++cur;
                ranks[cur] = r;
                nums[cur] += 1;
                river_hand_abstract[k][h][cur] = 1;
                hand2abstract[h] = cur;
            }
            else {
                nums[rank2abstract[r]] += 1;
                river_hand_abstract[k][h][rank2abstract[r]] = 1;
                hand2abstract[h] = rank2abstract[r];
            }
        }
        if (abs_count < cur+1)
            abs_count = cur+1;
        for (h=0; h<HAND_COUNT; ++h) {
            hand_1 = first_card[h];
            hand_2 = second_card[h];
            if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4)
                continue;

            equity_matrix_next_street[k][hand2abstract[h]][hand2abstract[h]] += 1;
            for (c=0; c<101; ++c) {
                if (hand_collide[h][c] <= h)
                    continue;
                hand_1 = first_card[hand_collide[h][c]];
                hand_2 = second_card[hand_collide[h][c]];
                if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                    hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4)
                    continue;
                equity_matrix_next_street[k][hand2abstract[h]][hand2abstract[hand_collide[h][c]]] += 1;
                equity_matrix_next_street[k][hand2abstract[hand_collide[h][c]]][hand2abstract[h]] += 1;
            }
        }

        for (int i=0; i<=cur; ++i) {
            fold_matrix_next_street[k][i][i] = 1 - equity_matrix_next_street[k][i][i] / nums[i] / nums[i];
            for (int j=i+1; j<=cur; ++j) {
                fold_matrix_next_street[k][i][j] = 1 - equity_matrix_next_street[k][i][j] / nums[i] / nums[j];
                fold_matrix_next_street[k][j][i] = fold_matrix_next_street[k][i][j];
            }
        }

        for (int i=0; i<=cur; ++i) {
            equity_matrix_next_street[k][i][i] = 0;
            for (int j=i+1; j<=cur; ++j)
                if (ranks[i] < ranks[j]) {
                    equity_matrix_next_street[k][i][j] = 1 - equity_matrix_next_street[k][i][j] / nums[i] / nums[j];
                    equity_matrix_next_street[k][j][i] = -equity_matrix_next_street[k][i][j];
                }
                else {
                    equity_matrix_next_street[k][i][j] = -1 + equity_matrix_next_street[k][i][j] / nums[i] / nums[j];
                    equity_matrix_next_street[k][j][i] = -equity_matrix_next_street[k][i][j];
                }
        }
    }
    return abs_count;
}

int set_river_abstract_combine_cpp(int board[4], float river_hand_abstract[][1326][149], float equity_matrix_next_street[][149], float fold_matrix_next_street[][149]) {
    int k = -1;
    int r, h, c, tmp;
    int ranks[149], nums[149];
    int rank2abstract[7463], hand2abstract[HAND_COUNT], hand_rank[HAND_COUNT];
    int hand_1, hand_2;
    int cur, abs_count = -1, board_abs_count = 0, valid_board_count = 0;
    int board_abs_counts[52];
    auto equity_matrix_next_street_each = new float[48][149][149];
    auto fold_matrix_next_street_each = new float[48][149][149];
    memset( equity_matrix_next_street_each, 0, 48*149*149*sizeof( float ) );
    memset( fold_matrix_next_street_each, 0, 48*149*149*sizeof( float ) );
    for (uint8_t board_4=0; board_4<52; ++board_4) {
        if (board_4 == board[0] || board_4 == board[1] || board_4 == board[2] || board_4 == board[3])
            continue;
        k ++;
        cur = -1;
        memset( rank2abstract, -1, sizeof( rank2abstract ) );
        memset( nums, 0, sizeof( nums ) );
        memset( hand2abstract, 0, sizeof( hand2abstract ) );
        memset( hand_rank, 0, sizeof( hand_rank ) );
        for (h=0; h<HAND_COUNT; ++h) {
            hand_1 = first_card[h];
            hand_2 = second_card[h];
            if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4)
                continue;
            r = SevenEval::GetRank(board[0], board[1], board[2], board[3], board_4, hand_1, hand_2);
            hand_rank[h] = r;
            if (rank2abstract[r] == -1) {
                rank2abstract[r] = ++cur;
                ranks[cur] = r;
                nums[cur] += 1;
            }
            else
                nums[rank2abstract[r]] += 1;
        }
        board_abs_count = cur + 1;
        board_abs_counts[board_4] = board_abs_count;
        for (int i=0; i<board_abs_count; ++i) {
            for (int j=i+1; j<board_abs_count; ++j) {
                if (ranks[i] > ranks[j])
                    continue;
                tmp = ranks[i];
                ranks[i] = ranks[j];
                ranks[j] = tmp;
                rank2abstract[ranks[i]] = i;
                rank2abstract[ranks[j]] = j;
                tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
            }
        }
        for (h=0; h<HAND_COUNT; ++h) {
            r = hand_rank[h];
            if (r == 0)
                continue;
            hand2abstract[h] = rank2abstract[r];
            river_hand_abstract[k][h][rank2abstract[r]] = 1;
        }

        for (h=0; h<HAND_COUNT; ++h) {
            hand_1 = first_card[h];
            hand_2 = second_card[h];
            if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4)
                continue;

            equity_matrix_next_street_each[k][hand2abstract[h]][hand2abstract[h]] += 1;
            for (c=0; c<101; ++c) {
                if (hand_collide[h][c] <= h)
                    continue;
                hand_1 = first_card[hand_collide[h][c]];
                hand_2 = second_card[hand_collide[h][c]];
                if (hand_1 == board[0] || hand_1 == board[1] || hand_1 == board[2] || hand_1 == board[3] || hand_1 == board_4 ||
                    hand_2 == board[0] || hand_2 == board[1] || hand_2 == board[2] || hand_2 == board[3] || hand_2 == board_4)
                    continue;
                equity_matrix_next_street_each[k][hand2abstract[h]][hand2abstract[hand_collide[h][c]]] += 1;
                equity_matrix_next_street_each[k][hand2abstract[hand_collide[h][c]]][hand2abstract[h]] += 1;
            }
        }

        for (int i=0; i<board_abs_count; ++i) {
            fold_matrix_next_street_each[k][i][i] = 1 - equity_matrix_next_street_each[k][i][i] / nums[i] / nums[i];
            for (int j=i+1; j<board_abs_count; ++j) {
                fold_matrix_next_street_each[k][i][j] = 1 - equity_matrix_next_street_each[k][i][j] / nums[i] / nums[j];
                fold_matrix_next_street_each[k][j][i] = fold_matrix_next_street_each[k][i][j];
            }
        }
        for (int i=0; i<board_abs_count; ++i) {
            equity_matrix_next_street_each[k][i][i] = 0;
            for (int j=i+1; j<board_abs_count; ++j)
            {
                equity_matrix_next_street_each[k][i][j] = -1 + equity_matrix_next_street_each[k][i][j] / nums[i] / nums[j];
                equity_matrix_next_street_each[k][j][i] = -equity_matrix_next_street_each[k][i][j];
            }
        }
        abs_count = max(abs_count, board_abs_count);
    }

    for (int i=0; i<abs_count; ++i) {
        equity_matrix_next_street[i][i] = 0;
        for (int j=i; j<abs_count; ++j) {
            k = -1;
            valid_board_count = 0;
            for (uint8_t board_4=0; board_4<52; ++board_4) {
                if (board_4 == board[0] || board_4 == board[1] || board_4 == board[2] || board_4 == board[3])
                    continue;
                k ++;
                if (board_abs_counts[board_4] <= j)
                    continue;
                valid_board_count ++;
                if (i == j)
                    fold_matrix_next_street[i][j] += fold_matrix_next_street_each[k][i][j];
                else {
                    equity_matrix_next_street[i][j] += equity_matrix_next_street_each[k][i][j];
                    equity_matrix_next_street[j][i] += equity_matrix_next_street_each[k][j][i];
                    fold_matrix_next_street[i][j] += fold_matrix_next_street_each[k][i][j];
                    fold_matrix_next_street[j][i] += fold_matrix_next_street_each[k][j][i];
                }
            }
            if (i == j)
                fold_matrix_next_street[i][j] /= valid_board_count;
            else {
                equity_matrix_next_street[i][j] /= valid_board_count;
                equity_matrix_next_street[j][i] /= valid_board_count;
                fold_matrix_next_street[i][j] /= valid_board_count;
                fold_matrix_next_street[j][i] /= valid_board_count;
            }
        }
    }

    free(equity_matrix_next_street_each);
    free(fold_matrix_next_street_each);
    return abs_count;
}

uint8_t comp_state(vector<uint8_t> &state, vector<uint8_t> &tmp) {
    for (int i = 0; i < state.size(); ++i)
    {
        if (state[i] < tmp[i])
            return 0;
        else if (state[i] > tmp[i])
            return 2;
    }
    return 1;
}

void get_min_state_5(vector<uint8_t> &state) {
    vector<uint8_t> tmp(state.size()), new_state(state);
    for (int i=1; i<24; ++i) {
        for (int j=0; j<state.size(); ++j)
            tmp[j] = card_premute[i][state[j]];
        sort(tmp.begin(), tmp.end()-2);
        sort(tmp.end()-2, tmp.end());
        if (comp_state(new_state, tmp) == 2)
            new_state.assign(tmp.begin(), tmp.end());
    }
    state.assign(new_state.begin(), new_state.end());
}

int get_state_id_in_3(int table_hand[5]) {
    uint8_t table_0 = (uint8_t)table_hand[0];
    uint8_t table_1 = (uint8_t)table_hand[1];
    uint8_t table_2 = (uint8_t)table_hand[2];
    uint8_t hand_1 = (uint8_t)table_hand[3];
    uint8_t hand_2 = (uint8_t)table_hand[4];
    vector<uint8_t> state = {table_0, table_1, table_2, hand_1, hand_2};
    sort(state.begin(), state.end()-2);
    sort(state.end()-2, state.end());
    get_min_state_5(state);

    // get_table_id
    uint32_t table_id = 0;
    table_id += table_3_num[state[0]];
    table_id += table_4_num[state[1]] - table_4_num[state[0]+1];
    table_id += state[2] - state[1] - 1;

    // get_hand_id
    int hand_1_id = 0, hand_1_num = 0, hand_2_num = 0;
    uint32_t hand_id = 0;
    for (int i=0; i<3; ++i) {
        if (state[3] < state[i])
            hand_1_num ++;
        if (state[4] < state[i])
            hand_2_num ++;
    }
    for (int i=0; i<state[3]; ++i) {
        if (i == state[hand_1_id])
            hand_1_id ++;
        else
            hand_id += 48 - i + hand_1_id;
    }
    hand_id += state[4] + hand_2_num - state[3] - hand_1_num - 1;

    return table_id * 1176 + hand_id;
}

int get_state_id_in_4(int table_hand[6]) {
    uint8_t table_0 = (uint8_t)table_hand[0];
    uint8_t table_1 = (uint8_t)table_hand[1];
    uint8_t table_2 = (uint8_t)table_hand[2];
    uint8_t table_3 = (uint8_t)table_hand[3];
    uint8_t hand_1 = (uint8_t)table_hand[4];
    uint8_t hand_2 = (uint8_t)table_hand[5];
    vector<uint8_t> state = {table_0, table_1, table_2, table_3, hand_1, hand_2};
    sort(state.begin(), state.end()-2);
    sort(state.end()-2, state.end());
    get_min_state_5(state);

    // get_table_id
    uint32_t table_id = 0;
    table_id += table_2_num[state[0]];
    table_id += table_3_num[state[1]] - table_3_num[state[0]+1];
    table_id += table_4_num[state[2]] - table_4_num[state[1]+1];
    table_id += state[3] - state[2] - 1;

    // get_hand_id
    int hand_1_id = 0, hand_1_num = 0, hand_2_num = 0;
    uint32_t hand_id = 0;
    for (int i=0; i<4; ++i) {
        if (state[4] < state[i])
            hand_1_num ++;
        if (state[5] < state[i])
            hand_2_num ++;
    }
    for (int i=0; i<state[4]; ++i) {
        if (i == state[hand_1_id])
            hand_1_id ++;
        else
            hand_id += 47 - i + hand_1_id;
    }
    hand_id += state[5] + hand_2_num - state[4] - hand_1_num - 1;

    return (int)(table_id * 1128 + hand_id);

}

bool floatLEQ(float a, float b) {
    return a - b < 1e-6;
}

void get_river_win_tie_cpp(int table[5], int hand_values[1326], int hand_win_tie[][1326]) {
    evaluate_hand_street_4_cpp(table, hand_values);
    int rank_count[7463], rank2win[7463];
    memset( rank_count, 0, sizeof( rank_count ) );
    memset( rank2win, 0, sizeof( rank2win ) );
    for (int i=0; i<1326; ++i)
        if (hand_values[i] > 0)
            rank_count[hand_values[i]] ++;
    int cum = 0;
    for (int i=0; i<7463; ++i)
        if (rank_count[i] > 0) {
            rank2win[i] = cum;
            cum += rank_count[i];
        }
    for (int i=0; i<1326; ++i) {
        if (hand_values[i] == 0)
            continue;
        hand_win_tie[0][i] = rank2win[hand_values[i]];
        hand_win_tie[1][i] = rank_count[hand_values[i]];
        for (int j=0; j<101; ++j) {
            int collide_value = hand_values[hand_collide[i][j]];
            if (collide_value == 0)
                continue;
            if (collide_value < hand_values[i])
                hand_win_tie[0][i] --;
            else if (collide_value == hand_values[i])
                hand_win_tie[1][i] --;
        }
    }
}

void load_hand2bucket_single_turn_cpp(int table[4], unsigned short board_abs_counts[48], short hand2abstract[][1326], unsigned short abs2hand[][1081], unsigned short abs_pos[][149], unsigned short abs_count[][149][2], unsigned short hand_collide_count[][1326][3]) {
    string folder = "/home/zhou/Project/PyProjects/deepstack/Data/turn_network_data/hand2bucket_cache";
    string slash = "/";
    string underline = "_";
    string pref = folder + slash + to_string(table[0]) + underline + to_string(table[1]) + slash + to_string(table[0]) + underline + to_string(table[1]) + underline +
                  to_string(table[2]) + underline + to_string(table[3]);

    string str = pref + "_board_abs_counts.bin";
    read_pointer((uint16_t *)board_abs_counts, str.c_str());

    str = pref + "_hand2abstract.bin";
    read_pointer((int16_t *)hand2abstract, str.c_str());

    str = pref + "_abs2hand.bin";
    read_pointer((uint16_t *)abs2hand, str.c_str());

    str = pref + "_abs_pos.bin";
    read_pointer((uint16_t *)abs_pos, str.c_str());

    str = pref + "_abs_count.bin";
    read_pointer((uint16_t *)abs_count, str.c_str());

    str = pref + "_hand_collide_count.bin";
    read_pointer((uint16_t *)hand_collide_count, str.c_str());
}

int toHash (const char* pString)
{
    int nSeed1, nSeed2;

    // LOL, coder joke: Dead Code ;)
    nSeed1 = 0xDEADC0DE;
    nSeed2 = 0x7FED7FED;

    const char* pKey = pString;
    char  ch;

//    while (*pKey != 0)
    for (int i=0; i<51; ++i)
    {
        ch = *pKey++;

        // if you changed the size of the cryptTable, you must change the & 0xFF below
        // to & whatever if it's a power of two, or % whatever, if it's not

        nSeed1 = cryptTable[((TYPE << 8) + ch)&0xFF] ^ (nSeed1 + nSeed2);
        nSeed2 = ch + nSeed1 + nSeed2 + (nSeed2 << 5) + 3;
    }

    return nSeed1;
}
