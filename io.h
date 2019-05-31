//
// Created by zhou on 19-5-29.
//

#ifndef DEEPSTACuint32_t_CPP_IO_H
#define DEEPSTACuint32_t_CPP_IO_H

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <fstream>
#include "constants.h"

using namespace std;

template <class T>
void write_vec(vector<T> &vec, const char *s) {
    ofstream f_write(s,ios::binary);
    auto size = (uint32_t)vec.size();
    f_write.write(reinterpret_cast<char *>(&size), sizeof(uint32_t));
    f_write.write(reinterpret_cast<char *>(&vec[0]), (uint64_t)size*sizeof(T));
    f_write.close();
}

template <class T>
void read_vec(vector<T> &vec, const char *s) {
    ifstream f_read(s, ios::binary);
    uint32_t size;
    f_read.read(reinterpret_cast<char *>(&size), sizeof(uint32_t));
    f_read.read(reinterpret_cast<char *>(&vec[0]), (uint64_t)size*sizeof(T));
    f_read.close();
}

template <class T>
void write_pointer(T *p, const char *s, uint32_t size) {
    ofstream f_write(s,ios::binary);
    f_write.write( reinterpret_cast<char *>(&size), sizeof(uint32_t) );
    f_write.write( reinterpret_cast<char *>(&p[0]), (uint64_t)size*sizeof(T) );
    f_write.close();
}

template <class T>
void read_pointer(T *p, const char *s) {
    ifstream f_read(s, ios::binary);
    uint32_t size;
    f_read.read( reinterpret_cast<char *>(&size),sizeof(uint32_t) );
    f_read.read( reinterpret_cast<char *>(&p[0]), (uint64_t)size*sizeof(T) );
    f_read.close();
}

#endif //DEEPSTACuint32_t_CPP_IO_H
