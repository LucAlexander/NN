#include "rng.h"
#include <time.h>
#include <stdlib.h>
#include <stdarg.h>

void startRandSeed(){
	srand(time(0));
}

int32_t randRange(int32_t a, int32_t b){
	return rand()%(b-a+1)+a;
}

float randRangeF(float a, float b){
	float range = b-a;
	float result = ((float)rand()/(float)RAND_MAX)*range;
	result += a;
	return result;
}

uint32_t rollDie(uint32_t sides){
	return randRange(1, sides);
}

int32_t randChoiceI(uint32_t n, ...){
	va_list args;
	va_start(args, n);
	uint32_t index = rollDie(n);
	int32_t val = -1;
	while(index > 0){
		val = va_arg(args, int);
		index--;
	}
	va_end(args);
	return val;
}

double randChoiceF(uint32_t n, ...){
	va_list args;
	va_start(args, n);
	uint32_t index = rollDie(n);
	double val = -1.0f;
	while(index > 0){
		val = va_arg(args, double);
		index--;
	}
	va_end(args);
	return val;
}

const char* randChoiceS(uint32_t n, ...){
	va_list args;
	va_start(args, n);
	uint32_t index = rollDie(n);
	const char* val = "";
	while(index > 0){
		val = va_arg(args, char*);
		index--;
	}
	va_end(args);
	return val;
}
