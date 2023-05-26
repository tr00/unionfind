#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "uf16.h"

#ifdef __AVX2__
#include <immintrin.h>

static const uint16_t T16L[] __attribute__((aligned(32))) =
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};

static const uint16_t T16H[] __attribute__((aligned(32))) =
    {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
     0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};

static const uint16_t R16[] __attribute__((aligned(32))) =
    {0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
     0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20};

static const uint32_t T32[] __attribute__((aligned(32))) =
    {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};

static const uint32_t R32[] __attribute__((aligned(32))) =
    {0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08};

/**
 * Gather 8x 16-bit integers from memory using 32-bit indices.
 * 16-bit elements are loaded from addresses starting at `addr`
 * and offset by each 32-bit index in `vi` scaled by 2.
 * The elements get zero extended and are stored as 32-bit
 * unsigned integers.
 */
static inline __m256i _mm256_u16gather_epi32(void *addr, __m256i vi)
{
    return _mm256_and_si256(_mm256_set1_epi32(0x0000ffff),
                            _mm256_i32gather_epi32(addr, vi, 2));
}

static inline int _mm256_testeq_epi8(__m256i a, __m256i b)
{
    return -1 == _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b));
}

/**
 * Stores the lower 128-bit vector lane of `vx` into memory.
 * `addr` does not need to be aligned.
 */
static inline void _mm256_storelo128_si256(void *addr, __m256i vx)
{
    _mm_storeu_si128(addr, _mm256_extracti128_si256(vx, 0));
}

#endif

void uf_init(uf_t *uf)
{
    uf->len = 0;
    uf->cap = 32;

    uf->vec = malloc(uf->cap * sizeof(uint16_t));

#ifdef __AVX2__

    _mm256_store_si256((__m256i *)&uf->vec[0x00], 
        _mm256_load_si256((__m256i *)&T16L));

    _mm256_store_si256((__m256i *)&uf->vec[0x10], 
        _mm256_load_si256((__m256i *)&T16H));

#else

    for (int i = 0; i < uf->cap; ++i)
        uf->vec[i] = i;

#endif
}

void uf_free(uf_t *uf)
{
    free(uf->vec);
}

uint16_t uf_push(uf_t *uf)
{
    if (__builtin_expect(uf->len == uf->cap, 0))
    {
        uf->cap *= 2;
        uf->vec = realloc(uf->vec, uf->cap * sizeof(uint16_t));

    #if 0

        __m256i vi, v0, v1, vx;

        vi = _mm256_set1_epi16(uf->len);

        v0 = _mm256_add_epi16(vi, _mm256_load_si256((__m256i *)&T16L));
        v1 = _mm256_add_epi16(vi, _mm256_load_si256((__m256i *)&T16H));

        vx = _mm256_load_si256((__m256i *)&R16);

        for (int i = uf->len; i < uf->cap; i += 32)
        {
            _mm256_store_si256((__m256i *)&uf->vec[i + 0x00], v0);
            _mm256_store_si256((__m256i *)&uf->vec[i + 0x10], v1);

            v0 = _mm256_add_epi16(v0, vx);
            v1 = _mm256_add_epi16(v1, vx);
        }

    #else

        for (int i = uf->len; i < uf->cap; ++i)
            uf->vec[i] = i;

    #endif
    }

    return (uint16_t)(uf->len++);
}

// =============== ROOT ================

uint16_t uf_root(uf_t *uf, uint16_t x)
{
    int16_t p;
    while ((p = uf->vec[x]) != x)
    {
        uf->vec[x] = uf->vec[p];
        x = p;
    }

    return x;
}

uint16_t uf_root_n(uf_t *uf, uint16_t x)
{
    return uf->vec[x];
}

#ifdef __AVX2__

__m256i uf_vroot(uf_t *uf, __m256i vx /* 8x uint32lo */)
{
    __m256i vp, vm;

    do
    {
        vp = _mm256_u16gather_epi32(uf->vec, vx);
        vm = _mm256_cmpeq_epi32(vp, vx);

        vx = vp;
    } while (_mm256_movemask_epi8(vm) != -1);

    return vx;
}

__m256i uf_vroot_n(uf_t *uf, __m256i vx)
{
    return _mm256_u16gather_epi32(uf->vec, vx);
}

#endif

// ================ JOIN ==================

uint16_t uf_join(uf_t *uf, uint16_t a, uint16_t b)
{
    uint16_t ra = uf_root(uf, a);
    uint16_t rb = uf_root(uf, b);

    if (ra == rb)
        return ra;

    if (rb < ra)
    {
        ra ^= rb;
        rb ^= ra;
        ra ^= rb;
    }

    uf->vec[rb] = ra;

    return ra;
}

// ================== NORM ==================

void uf_norm(uf_t *uf)
{
    for (int i = 0; i < uf->len; ++i)
    {
        uint16_t x = i;

        while (uf->vec[x] != x)
            x = uf->vec[x];

        uf->vec[i] = x;
    }
}

#ifdef __AVX2__

static const uint8_t SHF[] __attribute__((aligned(32))) = {
    0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
};

void uf_vnorm(uf_t *uf)
{
    __m256i vx, vp, vq, vs;

    vx = _mm256_load_si256((__m256i *)&T32);
    vs = _mm256_load_si256((__m256i *)&SHF);

    for (int i = 0; i < uf->len; i += 8)
    {
        vp = _mm256_u16gather_epi32(uf->vec, vx); /* parents */
        vq = _mm256_u16gather_epi32(uf->vec, vp); /* grannys */

        if (_mm256_testeq_epi8(vp, vq))
            continue;

        vx = vq;

        do {

            vp = _mm256_u16gather_epi32(uf->vec, vx);

            vq = vx;
            vx = vp;

        } while(!_mm256_testeq_epi8(vp, vq));


        // [ 0 | - | 2 | - | 4 | - | 6 | - | 8 | - | a | - | c | - | e | - ]
        // [ 0 | 2 | 4 | 6 | - | - | - | - | 8 | a | c | e | - | - | - | - ]
        vx = _mm256_shuffle_epi8(vx, vs);

        // [ 0 | 2 | 4 | 6 | - | - | - | - | 8 | a | c | e | - | - | - | - ]
        // [ 0 | 2 | 4 | 6 | 8 | a | c | e | - | - | - | - | - | - | - | - ]
        vx = _mm256_permute4x64_epi64(vx, 0b1000);

        _mm256_storelo128_si256(&uf->vec[i], vx);
    }
}

#endif