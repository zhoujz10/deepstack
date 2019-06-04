//
// Created by zhou on 19-5-29.
//

#ifndef DEEPSTACK_CPP_CONSTANTS_H
#define DEEPSTACK_CPP_CONSTANTS_H


#include <map>
#include <cstdint>
#include <string>
#include <torch/torch.h>

static const c10::Device device = c10::Device(c10::DeviceType::CUDA);

static const char *hand_collide_file = "data/hand_collide.npy";

static const char *flop_equity_matrix_dir = "data/equity_matrix_flop/";



const uint16_t card_hand_collide[52][51] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50},
    {0,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100},
    {1,51,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149},
    {2,52,101,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197},
    {3,53,102,150,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244},
    {4,54,103,151,198,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290},
    {5,55,104,152,199,245,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335},
    {6,56,105,153,200,246,291,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379},
    {7,57,106,154,201,247,292,336,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422},
    {8,58,107,155,202,248,293,337,380,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464},
    {9,59,108,156,203,249,294,338,381,423,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505},
    {10,60,109,157,204,250,295,339,382,424,465,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545},
    {11,61,110,158,205,251,296,340,383,425,466,506,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584},
    {12,62,111,159,206,252,297,341,384,426,467,507,546,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622},
    {13,63,112,160,207,253,298,342,385,427,468,508,547,585,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659},
    {14,64,113,161,208,254,299,343,386,428,469,509,548,586,623,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695},
    {15,65,114,162,209,255,300,344,387,429,470,510,549,587,624,660,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730},
    {16,66,115,163,210,256,301,345,388,430,471,511,550,588,625,661,696,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764},
    {17,67,116,164,211,257,302,346,389,431,472,512,551,589,626,662,697,731,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797},
    {18,68,117,165,212,258,303,347,390,432,473,513,552,590,627,663,698,732,765,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829},
    {19,69,118,166,213,259,304,348,391,433,474,514,553,591,628,664,699,733,766,798,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860},
    {20,70,119,167,214,260,305,349,392,434,475,515,554,592,629,665,700,734,767,799,830,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890},
    {21,71,120,168,215,261,306,350,393,435,476,516,555,593,630,666,701,735,768,800,831,861,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919},
    {22,72,121,169,216,262,307,351,394,436,477,517,556,594,631,667,702,736,769,801,832,862,891,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947},
    {23,73,122,170,217,263,308,352,395,437,478,518,557,595,632,668,703,737,770,802,833,863,892,920,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974},
    {24,74,123,171,218,264,309,353,396,438,479,519,558,596,633,669,704,738,771,803,834,864,893,921,948,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000},
    {25,75,124,172,219,265,310,354,397,439,480,520,559,597,634,670,705,739,772,804,835,865,894,922,949,975,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025},
    {26,76,125,173,220,266,311,355,398,440,481,521,560,598,635,671,706,740,773,805,836,866,895,923,950,976,1001,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049},
    {27,77,126,174,221,267,312,356,399,441,482,522,561,599,636,672,707,741,774,806,837,867,896,924,951,977,1002,1026,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072},
    {28,78,127,175,222,268,313,357,400,442,483,523,562,600,637,673,708,742,775,807,838,868,897,925,952,978,1003,1027,1050,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094},
    {29,79,128,176,223,269,314,358,401,443,484,524,563,601,638,674,709,743,776,808,839,869,898,926,953,979,1004,1028,1051,1073,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115},
    {30,80,129,177,224,270,315,359,402,444,485,525,564,602,639,675,710,744,777,809,840,870,899,927,954,980,1005,1029,1052,1074,1095,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135},
    {31,81,130,178,225,271,316,360,403,445,486,526,565,603,640,676,711,745,778,810,841,871,900,928,955,981,1006,1030,1053,1075,1096,1116,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154},
    {32,82,131,179,226,272,317,361,404,446,487,527,566,604,641,677,712,746,779,811,842,872,901,929,956,982,1007,1031,1054,1076,1097,1117,1136,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172},
    {33,83,132,180,227,273,318,362,405,447,488,528,567,605,642,678,713,747,780,812,843,873,902,930,957,983,1008,1032,1055,1077,1098,1118,1137,1155,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189},
    {34,84,133,181,228,274,319,363,406,448,489,529,568,606,643,679,714,748,781,813,844,874,903,931,958,984,1009,1033,1056,1078,1099,1119,1138,1156,1173,1190,1191,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205},
    {35,85,134,182,229,275,320,364,407,449,490,530,569,607,644,680,715,749,782,814,845,875,904,932,959,985,1010,1034,1057,1079,1100,1120,1139,1157,1174,1190,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220},
    {36,86,135,183,230,276,321,365,408,450,491,531,570,608,645,681,716,750,783,815,846,876,905,933,960,986,1011,1035,1058,1080,1101,1121,1140,1158,1175,1191,1206,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234},
    {37,87,136,184,231,277,322,366,409,451,492,532,571,609,646,682,717,751,784,816,847,877,906,934,961,987,1012,1036,1059,1081,1102,1122,1141,1159,1176,1192,1207,1221,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247},
    {38,88,137,185,232,278,323,367,410,452,493,533,572,610,647,683,718,752,785,817,848,878,907,935,962,988,1013,1037,1060,1082,1103,1123,1142,1160,1177,1193,1208,1222,1235,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259},
    {39,89,138,186,233,279,324,368,411,453,494,534,573,611,648,684,719,753,786,818,849,879,908,936,963,989,1014,1038,1061,1083,1104,1124,1143,1161,1178,1194,1209,1223,1236,1248,1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,1270},
    {40,90,139,187,234,280,325,369,412,454,495,535,574,612,649,685,720,754,787,819,850,880,909,937,964,990,1015,1039,1062,1084,1105,1125,1144,1162,1179,1195,1210,1224,1237,1249,1260,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280},
    {41,91,140,188,235,281,326,370,413,455,496,536,575,613,650,686,721,755,788,820,851,881,910,938,965,991,1016,1040,1063,1085,1106,1126,1145,1163,1180,1196,1211,1225,1238,1250,1261,1271,1281,1282,1283,1284,1285,1286,1287,1288,1289},
    {42,92,141,189,236,282,327,371,414,456,497,537,576,614,651,687,722,756,789,821,852,882,911,939,966,992,1017,1041,1064,1086,1107,1127,1146,1164,1181,1197,1212,1226,1239,1251,1262,1272,1281,1290,1291,1292,1293,1294,1295,1296,1297},
    {43,93,142,190,237,283,328,372,415,457,498,538,577,615,652,688,723,757,790,822,853,883,912,940,967,993,1018,1042,1065,1087,1108,1128,1147,1165,1182,1198,1213,1227,1240,1252,1263,1273,1282,1290,1298,1299,1300,1301,1302,1303,1304},
    {44,94,143,191,238,284,329,373,416,458,499,539,578,616,653,689,724,758,791,823,854,884,913,941,968,994,1019,1043,1066,1088,1109,1129,1148,1166,1183,1199,1214,1228,1241,1253,1264,1274,1283,1291,1298,1305,1306,1307,1308,1309,1310},
    {45,95,144,192,239,285,330,374,417,459,500,540,579,617,654,690,725,759,792,824,855,885,914,942,969,995,1020,1044,1067,1089,1110,1130,1149,1167,1184,1200,1215,1229,1242,1254,1265,1275,1284,1292,1299,1305,1311,1312,1313,1314,1315},
    {46,96,145,193,240,286,331,375,418,460,501,541,580,618,655,691,726,760,793,825,856,886,915,943,970,996,1021,1045,1068,1090,1111,1131,1150,1168,1185,1201,1216,1230,1243,1255,1266,1276,1285,1293,1300,1306,1311,1316,1317,1318,1319},
    {47,97,146,194,241,287,332,376,419,461,502,542,581,619,656,692,727,761,794,826,857,887,916,944,971,997,1022,1046,1069,1091,1112,1132,1151,1169,1186,1202,1217,1231,1244,1256,1267,1277,1286,1294,1301,1307,1312,1316,1320,1321,1322},
    {48,98,147,195,242,288,333,377,420,462,503,543,582,620,657,693,728,762,795,827,858,888,917,945,972,998,1023,1047,1070,1092,1113,1133,1152,1170,1187,1203,1218,1232,1245,1257,1268,1278,1287,1295,1302,1308,1313,1317,1320,1323,1324},
    {49,99,148,196,243,289,334,378,421,463,504,544,583,621,658,694,729,763,796,828,859,889,918,946,973,999,1024,1048,1071,1093,1114,1134,1153,1171,1188,1204,1219,1233,1246,1258,1269,1279,1288,1296,1303,1309,1314,1318,1321,1323,1325},
    {50,100,149,197,244,290,335,379,422,464,505,545,584,622,659,695,730,764,797,829,860,890,919,947,974,1000,1025,1049,1072,1094,1115,1135,1154,1172,1189,1205,1220,1234,1247,1259,1270,1280,1289,1297,1304,1310,1315,1319,1322,1324,1325},
};

const int max_abs_count = 149;

const int river_pots_fractions_05[20] = {1, 2, 3, 3, 6, 9, 9, 18, 27, 27, 54, 81, 81, 162, 243, 243, 486, 0, 0, 0};

const int river_pots_fractions_1[20] = {1, 2, 3, 6, 9, 18, 27, 54, 81, 162, 243, 486, 0, 0, 0, 0, 0, 0, 0, 0};

const float regret_epsilon = 1.0 / 1000000000;

const int boards_count[5] = { 0, 0, 22100, 49, 48 };

const bool pokermaster = false;

const float pot_fractions_by_street[5][2][3] = {
    { {   1, -1, -1 }, { 0.75, 1.25, -1 } },
    { { 0.5,  1, -1 }, {  0.5,    1,  2 } },
    { { 0.5,  1, -1 }, {    1,   -1, -1 } },
    { { 0.5,  1, -1 }, {    1,   -1, -1 } },
    { { 0.5,  1,  2 }, {  0.5,    1,  2 } },
};

#endif //DEEPSTACK_CPP_CONSTANTS_H
