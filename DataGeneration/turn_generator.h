//
// Created by zhou on 19-9-24.
//

#ifndef DEEPSTACK_CPP_TURN_GENERATOR_H
#define DEEPSTACK_CPP_TURN_GENERATOR_H

#include <torch/torch.h>
#include "../Game/board.h"
#include "RangeGenerator.h"
#include "../Lookahead/resolving.h"

void generate_turn_data(RangeGenerator &rg, const char *s);

#endif //DEEPSTACK_CPP_TURN_GENERATOR_H
