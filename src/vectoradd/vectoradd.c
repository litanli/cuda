#include <stdlib.h>

void vec_add(float* a, float* b, float* c, int n) {
    for (int i=0; i < n; i++)
        c[i] = a[i] + c[i];
}

int main() {

    // vec_add for n=200e6 took 0.43s (single core)
    int n = 200000000;
    float *a, *b, *c;
    a = (float*)malloc(n*sizeof(float));
    b = (float*)malloc(n*sizeof(float));
    c = (float*)malloc(n*sizeof(float));

    for (int i=0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;

    }

    vec_add(a, b, c, n);

    free(a);
    free(b);
    free(c);

    return 0;
}