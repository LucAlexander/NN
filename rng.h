#ifndef RANDNUMGEN_H
#define RANDNUMGEN_H

#include <inttypes.h>

void startRandSeed();

int32_t randRange(int32_t a, int32_t b);
float randRangeF(float a, float b);

uint32_t rollDie(uint32_t sides);

int32_t randChoiceI(uint32_t n, ...);
double randChoiceF(uint32_t n, ...);
const char* randChoiceS(uint32_t n, ...);

#endif
