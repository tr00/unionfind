#ifndef __UF16_H_
#define __UF16_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct
{
    int len;
    int cap;
    
    int32_t *vec;
} uf_t;

void uf_init(uf_t *uf);
void uf_free(uf_t *uf);

uint16_t uf_push(uf_t *uf);

uint16_t uf_root(uf_t *uf, uint16_t x);
uint16_t uf_root_n(uf_t *uf, uint16_t x);

bool uf_same(uf_t *uf, uint16_t a, uint16_t b);
bool uf_same_n(uf_t *uf, uint16_t a, uint16_t b);

uint16_t uf_join(uf_t *uf, uint16_t a, uint16_t b);

void uf_norm(uf_t *uf);


#endif
