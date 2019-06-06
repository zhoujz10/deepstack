//
// Created by zhou on 19-6-6.
//


#include "card_tools.h"

CardTools& get_card_tools() {
    static CardTools card_tools;
    return card_tools;
}