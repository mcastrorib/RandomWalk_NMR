#ifndef XORSHIFT_H_
#define XORSHIFT_H_

#include <iostream>
#include <stdint.h>

struct xorshift32_state
{
    uint32_t a;
};

uint32_t xorshift32(struct xorshift32_state *state);

struct xorshift64_state
{
    uint64_t a;
};

uint64_t xorshift64(struct xorshift64_state *state);

struct xorshift128_state
{
    uint32_t a, b, c, d;
};

uint32_t xorshift128(struct xorshift128_state *state);

// xorwow

struct xorwow_state
{
    uint32_t a, b, c, d;
    uint32_t counter;
};

uint32_t xoxwow(struct xorwow_state *state);

//xorshift*
struct xorshift64s_state
{
    uint64_t a;
};

uint64_t xorshift64s(struct xorshift64s_state *state);

/* The state must be seeded so that there is at least one non-zero element in array */
struct xorshift1024s_state
{
    uint64_t array[16];
    int index;
};

uint64_t xorshift1024s(struct xorshift1024s_state *state);

// xhorshift+
struct xorshift128p_state
{
    uint64_t a, b;
};

/* The state must be seeded so that it is not all zero */
uint64_t xorshift128p(struct xorshift128p_state *state);

// xoshiro256**
struct xoshiro256ss_state
{
    uint64_t s[4];
};

uint64_t rol64(uint64_t x, int k);

uint64_t xoshiro256ss(struct xoshiro256ss_state *state);

// xoshiro256+
struct xoshiro256p_state
{
    uint64_t s[4];
};

uint64_t xoshiro256p(struct xoshiro256p_state *state);

// SplitMix64
struct splitmix64_state
{
    uint64_t s;
};

uint64_t splitmix64(struct splitmix64_state *state);

struct xorshift128_state xorshift128_init(uint64_t seed);

double xorshift128plus_double(ulong *state0, ulong *state1);

// generate a random number using xorshift64
uint64_t generateRandomNumber(uint64_t _seed);

#endif