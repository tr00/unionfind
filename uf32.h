#ifndef __UFV2_H_
#define __UFV2_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct
{
    uint32_t *data;
    uint32_t *size;

    int len;
    int cap;
} uf_t;

void uf_init(uf_t *uf);
void uf_free(uf_t *uf);

uint32_t uf_push(uf_t *uf);

/**
 * Returns the canonical element for the set containing x.
 * To find out whether two elements belong to the same set compare roots.
 * 
 * Note: 
 *  This implementation uses path splitting to amortize runtime for
 *  future operations.
 *  A slight performance improvement of 5-25% can be made by removing
 *  the write instruction needed for path splitting. 
 *  However, those gains are mostly overshadowed by cache behaviour and
 *  for 1M or more elements and a ~50% merge factor amortization is a must.
 */
uint32_t uf_root(uf_t *uf, uint32_t x);

/**
 * A branchless version of uf_root() but works only after calling uf_norm()
 * and before calling uf_join() again.
 * 
 * Caller is responsible for assuring that the data structure is normalized.
 * 
 */
uint32_t uf_root_n(uf_t *uf, uint32_t x);

bool uf_same(uf_t *uf, uint32_t a, uint32_t b);

uint32_t uf_join(uf_t *uf, uint32_t a, uint32_t b);

void uf_norm(uf_t *uf);

#ifdef __AVX2__

#include <immintrin.h>

/**
 *
 * This is a vectorized implementation of uf_root() using AVX2.
 * Should give a speed up of at least 3x for sequential access.
 * 
 * Actual performance depends of this function is highly depended 
 * on the branch predictor!
 * 
 * Don't use this function for random access. Sequential only.
 * For random access use uf_norm() together with uf_vroot_n().
 * 
 */
static inline __m256i uf_vroot(uf_t *uf, __m256i vx)
{
    __m256i vp, vc, vm;

    vm = _mm256_set1_epi32(-1);

    do {

        vp = _mm256_mask_i32gather_epi32(vx, uf->data, vx, vm, 4);
        vc = _mm256_cmpeq_epi32(_mm256_setzero_si256(), vp);
        vm = _mm256_andnot_si256(vc, vm);
        vx = _mm256_blendv_epi8(vx, vp, vm);

    } while (_mm256_movemask_epi8(vm));

    return vx;
}

static inline __m256i uf_vroot_n(uf_t *uf, __m256i vx)
{
    __m256i vp, vc;

    vp = _mm256_i32gather_epi32(uf->data, vx, 4);
    vc = _mm256_cmpeq_epi32(_mm256_setzero_si256(), vp);
    vx = _mm256_blendv_epi8(vp, vx, vc);

    return vx;
}

#endif

#endif // __UF32_H_
