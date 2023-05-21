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

uint32_t uf_root(uf_t *uf, uint32_t x);

uint32_t uf_join(uf_t *uf, uint32_t a, uint32_t b);

void uf_norm(uf_t *uf);

#endif
