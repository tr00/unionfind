#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <immintrin.h>

#include "uf16.h"

void uf_init(uf_t *uf)
{
    uf->len = 0;
    uf->cap = 8;

    uf->vec = malloc(uf->cap * sizeof(int32_t));

    memset(uf->vec, -1, uf->cap * sizeof(int32_t));
}

void uf_free(uf_t *uf)
{
    free(uf->vec);
}

uint16_t uf_push(uf_t *uf)
{
    if (__builtin_expect(uf->len == uf->cap, false))
    {
        uf->cap *= 2;
        uf->vec = realloc(uf->vec, uf->cap * sizeof(int32_t));

        memset(&uf->vec[uf->len], -1, uf->len * sizeof(int32_t));
    }

    return (uint16_t)(uf->len++);
}

uint16_t uf_root(uf_t *uf, uint16_t x)
{
    while (uf->vec[x] >= 0)
        x = uf->vec[x];

    return x;
}

uint16_t uf_root_n(uf_t *uf, uint16_t x)
{
    int32_t p = uf->vec[x];

    return (uint16_t)(p >= 0 ? p : x);
}

uint16_t uf_join(uf_t *uf, uint16_t a, uint16_t b)
{
    uint16_t ra = uf_root(uf, a);
    uint16_t rb = uf_root(uf, b);

    if (ra == rb) 
        return rb;

    int32_t sa = -uf->vec[ra];
    int32_t sb = -uf->vec[rb];

    if (sa > sb)
    {
        ra ^= rb;
        rb ^= ra;
        ra ^= rb;
        
        sa ^= sb;
        sb ^= sa;
        sa ^= sb;
    }

    uf->vec[rb] -= sa;
    uf->vec[ra] = rb;

    return rb;
}

bool uf_same(uf_t *uf, uint16_t a, uint16_t b)
{
    return uf_root(uf, a) == uf_root(uf, b);
}

bool uf_same_n(uf_t *uf, uint16_t a, uint16_t b)
{
    return uf_root_n(uf, a) == uf_root_n(uf, b);
}

void uf_norm(uf_t *uf)
{
#ifdef __AVX2__

    static const int32_t offsets[] = {0, 1, 2, 3, 4, 5, 6, 7};

    __m256i vp, vm, vx, vc, vi, vq, v1;

    int len = (uf->len + 7) & ~7; // round up to multiple 8

    v1 = _mm256_set1_epi32(-1);
    vi = _mm256_loadu_si256((__m256i *)&offsets);

    for (int i = 0; i < len; i += 8)
    {
        vp = _mm256_add_epi32(_mm256_set1_epi32(i), vi);
        vm = _mm256_cmpgt_epi32(_mm256_set1_epi32(uf->len - i), vi);

        vx = vp;

        do {

            vq = _mm256_mask_i32gather_epi32(vx, uf->vec, vx, vm, 4);
            vc = _mm256_cmpgt_epi32(_mm256_setzero_si256(), vq);
            vm = _mm256_andnot_si256(vc, vm);
            vx = _mm256_blendv_epi8(vx, vq, vm);

        } while (_mm256_movemask_epi8(vm));

        vc = _mm256_cmpeq_epi32(vx, vp);
        vm = _mm256_xor_si256(vc, v1);

        _mm256_maskstore_epi32(&uf->vec[i], vm, vx);
    }

#else // fallback implementation

    int32_t p;

    for (int i = 0; i < uf->len; ++i)
    {
        p = uf_root(uf, i);

        if (p != i)
            uf->vec[i] = p;
    }
#endif
}