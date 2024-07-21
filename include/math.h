#ifndef MATH_H
#define MATH_H

namespace math {
    void identity(float* I, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                I[i * N + j] = i == j ? 1.0f : 0.0f;
            }
        }
    }

    void ones(float* mat, int N) {
        for (int i = 0; i < N * N; i++) {
            mat[i] = 1.0f;
        }
    }
}
#endif