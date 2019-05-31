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
#include <omp.h>
#include <time.h>
#include "hash/SevenEval.h"
#include "io.h"
#include "runtime_constants.h"

const int hand_idx[1326] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
        99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
        138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
        157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
        195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
        214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
        233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
        252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
        271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
        290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308,
        309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327,
        328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346,
        347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365,
        366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384,
        385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
        404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422,
        423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
        442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460,
        461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
        480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498,
        499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517,
        518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536,
        537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555,
        556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574,
        575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593,
        594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612,
        613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631,
        632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650,
        651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669,
        670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688,
        689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
        708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726,
        727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745,
        746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764,
        765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783,
        784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802,
        803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821,
        822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840,
        841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859,
        860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878,
        879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897,
        898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916,
        917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935,
        936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954,
        955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973,
        974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992,
        993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
        1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
        1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041,
        1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057,
        1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073,
        1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089,
        1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105,
        1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121,
        1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137,
        1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153,
        1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
        1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185,
        1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201,
        1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217,
        1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233,
        1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249,
        1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265,
        1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281,
        1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297,
        1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313,
        1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325};

//class MyHash {
//
//    public:
//
//        uint32_t hash_size;
//        int* hash_table;
//
//        MyHash() {
////            hash_size = 2147483648;
////            hash_table = new int[2147483648];
////            memset(hash_table, -1, (uint64_t)2147483648*sizeof(int));
//            hash_size = 1024;
//            hash_table = new int[1024];
//            memset(hash_table, -1, (uint64_t)1024*sizeof(int));
//        }
//
//};
//
//MyHash mh;

void quicksort(int length, int a[], int sorted_hand_value[]){
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

#endif //DEEPSTACK_CPP_RUNTIME_H
