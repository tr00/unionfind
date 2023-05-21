
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <immintrin.h>

#include "uf32.h"

void uf_init(uf_t *uf)
{
    uf->len = 1;
    uf->cap = 8;

    uf->data = calloc(uf->cap, sizeof(uint32_t));
    uf->size = calloc(uf->cap, sizeof(uint32_t));
}

void uf_free(uf_t *uf)
{
    free(uf->data);
    free(uf->size);
}

uint32_t uf_push(uf_t *uf)
{
    if (__builtin_expect(uf->len == uf->cap, false))
    {
        uf->cap *= 2;

        uf->data = realloc(uf->data, uf->cap * sizeof(uint32_t));
        uf->size = realloc(uf->size, uf->cap * sizeof(uint32_t));

        memset(&uf->data[uf->len], 0, uf->len * sizeof(uint32_t));
        memset(&uf->size[uf->len], 1, uf->len * sizeof(uint32_t));
    }

    return (uint32_t)(uf->len++);
}

uint32_t uf_root(uf_t *uf, uint32_t x)
{
#if defined(N)

    uint32_t p = uf->data[x];

    return p ? p : x;

#elif defined(A)

    uint32_t p;
    while ((p = uf->data[x]) != 0)
    {
        uf->data[x] = uf->data[p];
        x = p;
    }

    return x;


#else

    while (uf->data[x] != 0)
        x = uf->data[x];

    return x;

#endif
}

static uint32_t uf_root_i(uf_t *uf, uint32_t x)
{
#if defined(A)

    uint32_t p;
    while ((p = uf->data[x]) != 0)
    {
        uf->data[x] = uf->data[p];
        x = p;
    }

    return x;
#else

    while (uf->data[x] != 0)
        x = uf->data[x];

    return x;

#endif
}

uint32_t uf_join(uf_t *uf, uint32_t a, uint32_t b)
{
    uint32_t ra = uf_root_i(uf, a);
    uint32_t rb = uf_root_i(uf, b);

    if (ra == rb)
        return rb;

#ifdef S

    uint32_t sa = uf->size[ra];
    uint32_t sb = uf->size[rb];

    if (sa > sb)
    {
        ra ^= rb;
        rb ^= ra;
        ra ^= rb;
        
        sa ^= sb;
        sb ^= sa;
        sa ^= sb;
    }

    uf->size[rb] += sa;

#endif
    uf->data[ra]  = rb;

    return rb;
}

void uf_norm(uf_t *uf)
{
#ifdef N
#ifdef __AVX2__

    static const int32_t offsets[] = {0, 1, 2, 3, 4, 5, 6, 7};

    __m256i vp, vm, vx, vc, vi, v1, vo;

    v1 = _mm256_set1_epi32(-1);
    vo = _mm256_loadu_si256((__m256i *)&offsets);

    int len = (uf->len + 7) & ~7;

    for (int i = 0; i < len; i += 8)
    {
        vi = _mm256_add_epi32(_mm256_set1_epi32(i), vo);
        vm = _mm256_cmpgt_epi32(_mm256_set1_epi32(uf->len - i), vo);

        vx = vi;

        do {

            vp = _mm256_mask_i32gather_epi32(vx, uf->data, vx, vm, 4);
            vc = _mm256_cmpeq_epi32(_mm256_setzero_si256(), vp);
            vm = _mm256_andnot_si256(vc, vm);
            vx = _mm256_blendv_epi8(vx, vp, vm);

        } while (_mm256_movemask_epi8(vm));

        vc = _mm256_cmpeq_epi32(vx, vi);
        vm = _mm256_xor_si256(vc, v1);

        _mm256_maskstore_epi32((void *)&uf->data[i], vm, vx);
    }

#else

    uint32_t p;
    for (int i = 0; i < uf->len; ++i)
        if ((p = uf_root(uf, i)) != i)
            uf->data[i] = p;

#endif
#endif
}