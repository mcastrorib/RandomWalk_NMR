#include <stdint.h>

#include "xorshift.h"

/* The state word must be initialized to non-zero */
uint32_t xorshift32(struct xorshift32_state *state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = state->a;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state->a = x;
}

/* The state word must be initialized to non-zero */
uint64_t xorshift64(struct xorshift64_state *state)
{
    uint64_t x = state->a;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state->a = x;
}

/* The state array must be initialized to not be all zero */
uint32_t xorshift128(struct xorshift128_state *state)
{
    /* Algorithm "xor128" from p. 5 of Marsaglia, "Xorshift RNGs" */
    uint32_t t = state->d;

    uint32_t const s = state->a;
    state->d = state->c;
    state->c = state->b;
    state->b = s;

    t ^= t << 11;
    t ^= t >> 8;
    return state->a = t ^ s ^ (s >> 19);
}

// xorwow
/* The state array must be initialized to not be all zero in the first four words */
uint32_t xorwow(struct xorwow_state *state)
{
    /* Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs" */
    uint32_t t = state->d;

    uint32_t const s = state->a;
    state->d = state->c;
    state->c = state->b;
    state->b = s;

    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ (s << 4);
    state->a = t;

    state->counter += 362437;
    return t + state->counter;
}

// xorshift*

uint64_t xorshift64s(struct xorshift64s_state *state)
{
    uint64_t x = state->a; /* The state must be seeded with a nonzero value. */
    x ^= x >> 12;          // a
    x ^= x << 25;          // b
    x ^= x >> 27;          // c
    state->a = x;
    return x * UINT64_C(0x2545F4914F6CDD1D);
}

/* The state must be seeded so that there is at least one non-zero element in array */
uint64_t xorshift1024s(struct xorshift1024s_state *state)
{
    int index = state->index;
    uint64_t const s = state->array[index++];
    uint64_t t = state->array[index &= 15];
    t ^= t << 31;       // a
    t ^= t >> 11;       // b
    t ^= s ^ (s >> 30); // c
    state->array[index] = t;
    state->index = index;
    return t * (uint64_t)1181783497276652981;
}

// xorshift+
/* The state must be seeded so that it is not all zero */
uint64_t xorshift128p(struct xorshift128p_state *state)
{
    uint64_t t = state->a;
    uint64_t const s = state->b;
    state->a = s;
    t ^= t << 23;       // a
    t ^= t >> 17;       // b
    t ^= s ^ (s >> 26); // c
    state->b = t;
    return t + s;
}

// xoshiro256**
uint64_t rol64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256ss(struct xoshiro256ss_state *state)
{
    uint64_t *s = state->s;
    uint64_t const result = rol64(s[1] * 5, 7) * 9;
    uint64_t const t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rol64(s[3], 45);

    return result;
}
// xoshiro256p

uint64_t xoshiro256p(struct xoshiro256p_state *state)
{
    //uint64_t (*s)[4] = &state->s;
    uint64_t s[4];
    s[0] = state->s[0];
    s[1] = state->s[1];
    s[2] = state->s[2];
    s[3] = state->s[3];
    uint64_t const result = s[0] + s[3];
    uint64_t const t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rol64(s[3], 45);

    return result;
}

// SplitMix64
uint64_t splitmix64(struct splitmix64_state *state)
{
    uint64_t result = state->s;

    state->s = result + 0x9E3779B97f4A7C15;
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

// as an example; one could do this same thing for any of the other generators
struct xorshift128_state xorshift128_init(uint64_t seed)
{
    struct splitmix64_state smstate = {seed};
    struct xorshift128_state result = {0};

    uint64_t tmp = splitmix64(&smstate);
    result.a = (uint32_t)tmp;
    result.b = (uint32_t)(tmp >> 32);

    tmp = splitmix64(&smstate);
    result.c = (uint32_t)tmp;
    result.d = (uint32_t)(tmp >> 32);

    return result;
}

double xorshift128plus_double(ulong *state0, ulong *state1)
{
    ulong xor0 = *state0;     
    ulong xor1 = *state1;
    *state0 = xor1;  
    xor0 ^= xor0 << 23; 
    *state1 = xor0 ^ xor1 ^ (xor0 >> 17) ^ (xor1 >> 26);
    const ulong result = 0x3ff0000000000000 | ((*state1 + xor1) >> 12);  //IEEE 754 double between [1.0, 2.0);
    double* d_ptr = (double*)(&result); //reinterpreting 64bit integer to double 
    return *d_ptr - 1.0;   //-1.0 so the result is between [0.0, 1.0).       
}

// generate a random number using xorshift64
uint64_t generateRandomNumber(uint64_t _seed)
{
    xorshift64_state xor_state;
    xor_state.a = _seed;
    return xorshift64(&xor_state);
}
