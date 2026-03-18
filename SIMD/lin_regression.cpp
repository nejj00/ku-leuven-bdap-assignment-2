/*
 * Copyright 2022 BDAP team.
 *
 * Author: Laurens Devos
 * Version: 0.1
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <immintrin.h>

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

/**
 * A matrix representation.
 *
 * Based on:
 * https://github.com/laudv/veritas/blob/main/src/cpp/basics.hpp#L39
 */
template <typename T>
struct matrix {
private:
    std::vector<T> vec_;

public:
    size_t nrows, ncols;
    size_t stride_row, stride_col; // in num of elems, not bytes

    /** Compute the index of an element. */
    inline size_t index(size_t row, size_t col) const
    {
        if (row >= nrows)
            throw std::out_of_range("out of bounds row");
        if (col >= ncols)
            throw std::out_of_range("out of bounds column");
        return row * stride_row + col * stride_col;
    }

    /** Get a pointer to the data */
    inline const T *ptr() const { return vec_.data(); }

    /** Get a pointer to an element */
    inline const T *ptr(size_t row, size_t col) const
    { return &ptr()[index(row, col)]; }

    /** Get a pointer to the data */
    inline T *ptr_mut() { return vec_.data(); }

    /** Get a pointer to an element */
    inline T *ptr_mut(size_t row, size_t col)
    { return &ptr_mut()[index(row, col)]; }

    /** Access element in data matrix without bounds checking. */
    inline T get_elem(size_t row, size_t col) const
    { return ptr()[index(row, col)]; }

    /** Access element in data matrix without bounds checking. */
    inline void set_elem(size_t row, size_t col, T&& value)
    { ptr_mut()[index(row, col)] = std::move(value); }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](size_t i) const
    { return ptr()[i]; }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T& operator[](size_t i)
    { return ptr_mut()[i]; }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](std::pair<size_t, size_t> p) const
    { auto &&[i, j] = p; return get_elem(i, j); }

    matrix(std::vector<T>&& vec, size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(std::move(vec))
        , nrows(nr)
        , ncols(nc)
        , stride_row(sr)
        , stride_col(sc) {}

    matrix(size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(nr * nc)
        , nrows(nr)
        , ncols(nc)
        , stride_row(sr)
        , stride_col(sc) {}
};

using fmatrix = matrix<float>;

std::tuple<fmatrix, fmatrix, fmatrix, float>
read_bin_data(const char *fname)
{
    std::ifstream f(fname, std::ios::binary);
    
    if (!f) { throw std::runtime_error("opening file failed"); }

    char buf[8];
    f.read(buf, 8);

    int num_ex = *reinterpret_cast<int *>(&buf[0]);
    int num_feat = *reinterpret_cast<int *>(&buf[4]);

    std::cout << "num_ex " << num_ex << ", num_feat " << num_feat << std::endl;

    size_t num_numbers = num_ex * num_feat;
    fmatrix x(num_ex, num_feat, num_feat, 1);
    fmatrix y(num_ex, 1, 1, 1);
    fmatrix coef(num_feat, 1, 1, 1);

    f.read(reinterpret_cast<char *>(x.ptr_mut()), num_numbers * sizeof(float));
    f.read(reinterpret_cast<char *>(y.ptr_mut()), num_ex * sizeof(float));
    f.read(reinterpret_cast<char *>(coef.ptr_mut()), num_feat * sizeof(float));

    f.read(buf, sizeof(float));
    float intercept = *reinterpret_cast<float *>(&buf[0]);

    return std::make_tuple(x, y, coef, intercept);
}

fmatrix evaluate_scalar(const fmatrix& x, const fmatrix& y,
                        const fmatrix& coef, float intercept) {
    fmatrix output(x.nrows, 1, 1, 1);

    // TODO implement this method using regular C++
    for (size_t i = 0; i < x.nrows; i++)
     {
        float sum = intercept;

        for (size_t j = 0; j < x.ncols; j++) {
            sum += x.get_elem(i, j) * coef.get_elem(j, 0);
        }
        
        output.set_elem(i, 0, std::move(sum));
    }

    //
    // You CANNOT use threads.
    // We are forbidding multithreading to make the coding take less time and
    // get you to focus on learning the SIMD part.

    return output;
}

fmatrix evaluate_simd(const fmatrix& x, const fmatrix& y,
                        const fmatrix& coef, float intercept) {
    fmatrix output(x.nrows, 1, 1, 1);

    // TODO implement this method using SIMD intrinsic functions. See the second
    // exercise session.
    
    for (size_t i = 0; i < x.nrows; i++)
    {
        __m256 sum_vec = _mm256_setzero_ps();

        size_t j = 0;

        // Process 8 elements per iteration
        for (; j + 7 < x.ncols; j += 8)
        {
            __m256 x_vec    = _mm256_loadu_ps(x.ptr(i, j));
            __m256 coef_vec = _mm256_loadu_ps(coef.ptr(j, 0));

            sum_vec = _mm256_fmadd_ps(x_vec, coef_vec, sum_vec);
        }

        // Horizontal reduction of 8 floats in sum_vec
        __m128 low  = _mm256_castps256_ps128(sum_vec);
        __m128 high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(low, high);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float final_sum = _mm_cvtss_f32(sum128);

        // Handle remaining elements
        for (; j < x.ncols; j++)
        {
            final_sum += x.get_elem(i, j) * coef.get_elem(j, 0);
        }

        final_sum += intercept;

        output.set_elem(i, 0, std::move(final_sum));
    }

    //
    // You CANNOT use threads.
    // We are forbidding multithreading to make the coding take less time and
    // get you to focus on learning the SIMD part.

    return output;
}


fmatrix evaluate_simd_faster(const fmatrix& x, const fmatrix& y,
                        const fmatrix& coef, float intercept) {
    fmatrix output(x.nrows, 1, 1, 1);

    size_t ncols = x.ncols;
    size_t nrows = x.nrows;

    // Helper: horizontally reduce __m256 to a single float
    auto hsum = [](const __m256& v) -> float {
        __m128 low  = _mm256_castps256_ps128(v);
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 s    = _mm_add_ps(low, high);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };

    if (ncols >= 32) {
        // ── Strategy A: many features — 8 rows × 4 accumulators ──────────
        // Amortizes coefficient loads across 8 rows, hides FMA latency
        size_t i = 0;
        for (; i + 7 < nrows; i += 8) {
            __m256 acc0a = _mm256_setzero_ps(), acc0b = _mm256_setzero_ps(),
                   acc0c = _mm256_setzero_ps(), acc0d = _mm256_setzero_ps();
            __m256 acc1a = _mm256_setzero_ps(), acc1b = _mm256_setzero_ps(),
                   acc1c = _mm256_setzero_ps(), acc1d = _mm256_setzero_ps();
            __m256 acc2a = _mm256_setzero_ps(), acc2b = _mm256_setzero_ps(),
                   acc2c = _mm256_setzero_ps(), acc2d = _mm256_setzero_ps();
            __m256 acc3a = _mm256_setzero_ps(), acc3b = _mm256_setzero_ps(),
                   acc3c = _mm256_setzero_ps(), acc3d = _mm256_setzero_ps();
            __m256 acc4a = _mm256_setzero_ps(), acc4b = _mm256_setzero_ps(),
                   acc4c = _mm256_setzero_ps(), acc4d = _mm256_setzero_ps();
            __m256 acc5a = _mm256_setzero_ps(), acc5b = _mm256_setzero_ps(),
                   acc5c = _mm256_setzero_ps(), acc5d = _mm256_setzero_ps();
            __m256 acc6a = _mm256_setzero_ps(), acc6b = _mm256_setzero_ps(),
                   acc6c = _mm256_setzero_ps(), acc6d = _mm256_setzero_ps();
            __m256 acc7a = _mm256_setzero_ps(), acc7b = _mm256_setzero_ps(),
                   acc7c = _mm256_setzero_ps(), acc7d = _mm256_setzero_ps();

            size_t j = 0;
            for (; j + 31 < ncols; j += 32) {
                __m256 c0 = _mm256_loadu_ps(coef.ptr(j,      0));
                __m256 c1 = _mm256_loadu_ps(coef.ptr(j + 8,  0));
                __m256 c2 = _mm256_loadu_ps(coef.ptr(j + 16, 0));
                __m256 c3 = _mm256_loadu_ps(coef.ptr(j + 24, 0));

                acc0a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i,   j     )), c0, acc0a);
                acc0b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i,   j + 8 )), c1, acc0b);
                acc0c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i,   j + 16)), c2, acc0c);
                acc0d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i,   j + 24)), c3, acc0d);

                acc1a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+1, j     )), c0, acc1a);
                acc1b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+1, j + 8 )), c1, acc1b);
                acc1c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+1, j + 16)), c2, acc1c);
                acc1d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+1, j + 24)), c3, acc1d);

                acc2a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+2, j     )), c0, acc2a);
                acc2b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+2, j + 8 )), c1, acc2b);
                acc2c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+2, j + 16)), c2, acc2c);
                acc2d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+2, j + 24)), c3, acc2d);

                acc3a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+3, j     )), c0, acc3a);
                acc3b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+3, j + 8 )), c1, acc3b);
                acc3c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+3, j + 16)), c2, acc3c);
                acc3d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+3, j + 24)), c3, acc3d);

                acc4a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+4, j     )), c0, acc4a);
                acc4b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+4, j + 8 )), c1, acc4b);
                acc4c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+4, j + 16)), c2, acc4c);
                acc4d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+4, j + 24)), c3, acc4d);

                acc5a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+5, j     )), c0, acc5a);
                acc5b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+5, j + 8 )), c1, acc5b);
                acc5c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+5, j + 16)), c2, acc5c);
                acc5d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+5, j + 24)), c3, acc5d);

                acc6a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+6, j     )), c0, acc6a);
                acc6b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+6, j + 8 )), c1, acc6b);
                acc6c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+6, j + 16)), c2, acc6c);
                acc6d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+6, j + 24)), c3, acc6d);

                acc7a = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+7, j     )), c0, acc7a);
                acc7b = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+7, j + 8 )), c1, acc7b);
                acc7c = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+7, j + 16)), c2, acc7c);
                acc7d = _mm256_fmadd_ps(_mm256_loadu_ps(x.ptr(i+7, j + 24)), c3, acc7d);
            }

            // Merge 4 accumulators per row
            __m256 s0 = _mm256_add_ps(_mm256_add_ps(acc0a, acc0b), _mm256_add_ps(acc0c, acc0d));
            __m256 s1 = _mm256_add_ps(_mm256_add_ps(acc1a, acc1b), _mm256_add_ps(acc1c, acc1d));
            __m256 s2 = _mm256_add_ps(_mm256_add_ps(acc2a, acc2b), _mm256_add_ps(acc2c, acc2d));
            __m256 s3 = _mm256_add_ps(_mm256_add_ps(acc3a, acc3b), _mm256_add_ps(acc3c, acc3d));
            __m256 s4 = _mm256_add_ps(_mm256_add_ps(acc4a, acc4b), _mm256_add_ps(acc4c, acc4d));
            __m256 s5 = _mm256_add_ps(_mm256_add_ps(acc5a, acc5b), _mm256_add_ps(acc5c, acc5d));
            __m256 s6 = _mm256_add_ps(_mm256_add_ps(acc6a, acc6b), _mm256_add_ps(acc6c, acc6d));
            __m256 s7 = _mm256_add_ps(_mm256_add_ps(acc7a, acc7b), _mm256_add_ps(acc7c, acc7d));

            // Scalar remainder + intercept for each row
            auto finish = [&](size_t row, float partial) -> float {
                for (size_t jj = j; jj < ncols; jj++)
                    partial += x.get_elem(row, jj) * coef.get_elem(jj, 0);
                return partial + intercept;
            };

            output.set_elem(i,   0, finish(i,   hsum(s0)));
            output.set_elem(i+1, 0, finish(i+1, hsum(s1)));
            output.set_elem(i+2, 0, finish(i+2, hsum(s2)));
            output.set_elem(i+3, 0, finish(i+3, hsum(s3)));
            output.set_elem(i+4, 0, finish(i+4, hsum(s4)));
            output.set_elem(i+5, 0, finish(i+5, hsum(s5)));
            output.set_elem(i+6, 0, finish(i+6, hsum(s6)));
            output.set_elem(i+7, 0, finish(i+7, hsum(s7)));
        }

        // Remaining rows with single-row SIMD
        for (; i < nrows; i++) {
            __m256 acc = _mm256_setzero_ps();
            size_t j = 0;
            for (; j + 7 < ncols; j += 8) {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.ptr(i, j)),
                    _mm256_loadu_ps(coef.ptr(j, 0)),
                    acc);
            }
            float result = hsum(acc);
            for (; j < ncols; j++)
                result += x.get_elem(i, j) * coef.get_elem(j, 0);
            output.set_elem(i, 0, result + intercept);
        }

    } else {
        // ── Strategy B: few features (< 32) — simple single-row SIMD ─────
        // The multi-row strategy adds too much overhead for short rows.
        // Just vectorize the dot product per row with 2 accumulators
        // to hide FMA latency without excess register pressure.
        for (size_t i = 0; i < nrows; i++) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            size_t j = 0;
            for (; j + 15 < ncols; j += 16) {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.ptr(i, j)),
                    _mm256_loadu_ps(coef.ptr(j, 0)),
                    acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.ptr(i, j + 8)),
                    _mm256_loadu_ps(coef.ptr(j + 8, 0)),
                    acc1);
            }
            for (; j + 7 < ncols; j += 8) {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.ptr(i, j)),
                    _mm256_loadu_ps(coef.ptr(j, 0)),
                    acc0);
            }

            float result = hsum(_mm256_add_ps(acc0, acc1));
            for (; j < ncols; j++)
                result += x.get_elem(i, j) * coef.get_elem(j, 0);
            output.set_elem(i, 0, result + intercept);
        }
    }

    return output;
}



// int main(int argc, char *argv[])
// {
//     // These are four linear regression models
//     auto &&[x, y, coef, intercept] = read_bin_data("data/calhouse.bin");
//     // auto &&[x, y, coef, intercept] = read_bin_data("data/allstate.bin");
//     //auto &&[x, y, coef, intercept] = read_bin_data("data/diamonds.bin");
//     //auto &&[x, y, coef, intercept] = read_bin_data("data/cpusmall.bin");

//     // This is a logistic regression model, but can be evaluated in the same way
//     // All you would need to do is apply the sigmoid to the values in `output_*`
//     //auto &&[x, y, coef, intercept] = read_bin_data("data/mnist_5vall.bin");
    
//     // TODO repeat the number of time measurements to get a more accurate
//     // estimate of the runtime.

//     steady_clock::time_point tbegin, tend;

//     // SCALAR
//     tbegin = steady_clock::now();
//     auto output_scalar = evaluate_scalar(x, y, coef, intercept);
//     tend = steady_clock::now();

//     std::cout << "Evaluated scalar in "
//         << (duration_cast<microseconds>(tend-tbegin).count()/1000.0)
//         << "ms" << std::endl;

//     // SIMD
//     tbegin = steady_clock::now();
//     auto output_simd = evaluate_simd(x, y, coef, intercept);
//     tend = steady_clock::now();

//     std::cout << "Evaluated SIMD in "
//         << (duration_cast<microseconds>(tend-tbegin).count()/1000.0)
//         << "ms" << std::endl;

//     // TODO check output
// }

int main(int argc, char *argv[])
{
    const char* datasets[] = {
        "data/calhouse.bin",
        "data/allstate.bin",
        "data/diamonds.bin",
        "data/cpusmall.bin",
        "data/mnist_5vall.bin"
    };

    const int NUM_RUNS = 100;

    for (const char* dataset : datasets) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Dataset: " << dataset << std::endl;
        std::cout << "========================================" << std::endl;

        fmatrix x(0, 0, 0, 0), y(0, 0, 0, 0), coef(0, 0, 0, 0);
        float intercept = 0.0f;

        try {
            auto &&[x_, y_, coef_, intercept_] = read_bin_data(dataset);
            x = std::move(x_);
            y = std::move(y_);
            coef = std::move(coef_);
            intercept = intercept_;
        } catch (const std::exception& e) {
            std::cout << "Skipping — " << e.what() << std::endl;
            continue;
        }

        double scalar_total = 0.0;
        fmatrix output_scalar(0, 0, 0, 0);
        for (int run = 0; run < NUM_RUNS; run++) {
            auto tbegin = steady_clock::now();
            output_scalar = evaluate_scalar(x, y, coef, intercept);
            auto tend = steady_clock::now();
            scalar_total += duration_cast<microseconds>(tend - tbegin).count() / 1000.0;
        }
        double scalar_avg = scalar_total / NUM_RUNS;

        double simd_total = 0.0;
        fmatrix output_simd(0, 0, 0, 0);
        for (int run = 0; run < NUM_RUNS; run++) {
            auto tbegin = steady_clock::now();
            output_simd = evaluate_simd_faster(x, y, coef, intercept);
            auto tend = steady_clock::now();
            simd_total += duration_cast<microseconds>(tend - tbegin).count() / 1000.0;
        }
        double simd_avg = simd_total / NUM_RUNS;

        std::cout << "Scalar avg (" << NUM_RUNS << " runs): " << scalar_avg << "ms" << std::endl;
        std::cout << "SIMD   avg (" << NUM_RUNS << " runs): " << simd_avg   << "ms" << std::endl;
        std::cout << "Speedup:    " << (scalar_avg / simd_avg) << "x" << std::endl;
    }

    return 0;
}