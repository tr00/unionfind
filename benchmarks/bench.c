#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "uf32.h"

#undef N

#define N 1024 * 1024
#define M  768 * 1024
#define R 250

int main()
{
    clock_t c0, c1;

    uf_t uf;

    uf_init(&uf);

    for (int n = 0; n < N; ++n)
        uf_push(&uf);

    double mean = 0.0;

    uint64_t o = 0;

    for (int r = 0; r < R; ++r)
    {
        srand(r);

        // prepare
        for (int i = 0; i < M; ++i)
        {
            uint32_t a = rand() % (N - 1) + 1;
            uint32_t b = rand() % (N - 1) + 1;

            uf_join(&uf, a, b);
        }

        c0 = clock();

        uf_norm(&uf);

        // benchmark
        for (int i = 1; i < N; ++i)
            uf_root(&uf, rand() % (N - 1) + 1);

        c1 = clock();

        mean += (double)(c1 - c0) / CLOCKS_PER_SEC * 1000 * 1000;

        // reset
        memset(uf.data, 0, uf.cap * sizeof(uint32_t));
        memset(uf.size, 1, uf.cap * sizeof(uint32_t));

    }

    printf("mean time: %fus\n", mean / (double)R);

    uf_free(&uf);

    return 0;
}