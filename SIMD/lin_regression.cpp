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
#include <cmath>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

/**
 * A matrix representation.
 *
 * Based on:
 * https://github.com/laudv/veritas/blob/main/src/cpp/basics.hpp#L39
 */
template <typename T>
struct matrix
{
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
    {
        return &ptr()[index(row, col)];
    }

    /** Get a pointer to the data */
    inline T *ptr_mut() { return vec_.data(); }

    /** Get a pointer to an element */
    inline T *ptr_mut(size_t row, size_t col)
    {
        return &ptr_mut()[index(row, col)];
    }

    /** Access element in data matrix without bounds checking. */
    inline T get_elem(size_t row, size_t col) const
    {
        return ptr()[index(row, col)];
    }

    /** Access element in data matrix without bounds checking. */
    inline void set_elem(size_t row, size_t col, T &&value)
    {
        ptr_mut()[index(row, col)] = std::move(value);
    }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](size_t i) const
    {
        return ptr()[i];
    }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T &operator[](size_t i)
    {
        return ptr_mut()[i];
    }

    /** Access elements linearly (e.g. for when data is vector). */
    inline T operator[](std::pair<size_t, size_t> p) const
    {
        auto &&[i, j] = p;
        return get_elem(i, j);
    }

    matrix(std::vector<T> &&vec, size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(std::move(vec)), nrows(nr), ncols(nc), stride_row(sr), stride_col(sc) {}

    matrix(size_t nr, size_t nc, size_t sr, size_t sc)
        : vec_(nr * nc), nrows(nr), ncols(nc), stride_row(sr), stride_col(sc) {}
};

using fmatrix = matrix<float>;

std::tuple<fmatrix, fmatrix, fmatrix, float>
read_bin_data(const char *fname)
{
    std::ifstream f(fname, std::ios::binary);

    if (!f)
    {
        throw std::runtime_error("opening file failed");
    }

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

fmatrix evaluate_scalar(const fmatrix &x, const fmatrix &y,
                        const fmatrix &coef, float intercept)
{
    fmatrix output(x.nrows, 1, 1, 1);

    // TODO implement this method using regular C++
    for (size_t i = 0; i < x.nrows; i++)
    {
        float sum = intercept;

        for (size_t j = 0; j < x.ncols; j++)
        {
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

fmatrix evaluate_simd(const fmatrix &x, const fmatrix &y,
                             const fmatrix &coef, float intercept)
{
    fmatrix output(x.nrows, 1, 1, 1);

    const float *coef_ptr = coef.ptr(0, 0);
    size_t ncols = x.ncols;

    for (size_t i = 0; i < x.nrows; i++)
    {
        const float *x_row = x.ptr(i, 0);

        float final_sum = 0.0f;
        size_t j = 0;

        if (ncols < 32)
        {
            __m256 sum_vec = _mm256_setzero_ps();

            for (; j + 7 < ncols; j += 8)
            {
                sum_vec = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row + j),
                    _mm256_loadu_ps(coef_ptr + j),
                    sum_vec);
            }

            // reduction
            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);

            final_sum = _mm_cvtss_f32(sum128);
        }
        else
        {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();

            for (; j + 31 < ncols; j += 32)
            {
                sum0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row + j),
                    _mm256_loadu_ps(coef_ptr + j),
                    sum0);

                sum1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row + j + 8),
                    _mm256_loadu_ps(coef_ptr + j + 8),
                    sum1);

                sum2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row + j + 16),
                    _mm256_loadu_ps(coef_ptr + j + 16),
                    sum2);

                sum3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row + j + 24),
                    _mm256_loadu_ps(coef_ptr + j + 24),
                    sum3);
            }

            __m256 sum_vec = _mm256_add_ps(
                _mm256_add_ps(sum0, sum1),
                _mm256_add_ps(sum2, sum3));

            for (; j + 7 < ncols; j += 8)
            {
                sum_vec = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row + j),
                    _mm256_loadu_ps(coef_ptr + j),
                    sum_vec);
            }

            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);

            final_sum = _mm_cvtss_f32(sum128);
        }

        for (; j < ncols; j++)
        {
            final_sum += x_row[j] * coef_ptr[j];
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

int main(int argc, char *argv[])
{
    const char *datasets[] = {
        "/cw/bdap/assignment2/simd/calhouse.bin",
        "/cw/bdap/assignment2/simd/cpusmall.bin",
        "/cw/bdap/assignment2/simd/diamonds.bin",
        "/cw/bdap/assignment2/simd/allstate.bin",
        "/cw/bdap/assignment2/simd/mnist_5vall.bin"
    };
    
    const int NUM_RUNS = 100;

    for (const char *dataset : datasets)
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Dataset: " << dataset << std::endl;
        std::cout << "========================================" << std::endl;

        fmatrix x(0, 0, 0, 0), y(0, 0, 0, 0), coef(0, 0, 0, 0);
        float intercept = 0.0f;

        try
        {
            auto &&[x_, y_, coef_, intercept_] = read_bin_data(dataset);
            x = std::move(x_);
            y = std::move(y_);
            coef = std::move(coef_);
            intercept = intercept_;
        }
        catch (const std::exception &e)
        {
            std::cout << "Skipping — " << e.what() << std::endl;
            continue;
        }

        std::vector<double> scalar_times;
        double scalar_total = 0.0;
        fmatrix output_scalar(0, 0, 0, 0);
        for (int run = 0; run < NUM_RUNS; run++)
        {
            auto tbegin = steady_clock::now();
            output_scalar = evaluate_scalar(x, y, coef, intercept);
            auto tend = steady_clock::now();

            double time = duration_cast<microseconds>(tend - tbegin).count() / 1000.0;
            scalar_times.push_back(time);
            scalar_total += time;
        }
        double scalar_avg = scalar_total / NUM_RUNS;
        double scalar_var = 0.0;
        for (double t : scalar_times)
        {
            scalar_var += (t - scalar_avg) * (t - scalar_avg);
        }
        scalar_var /= NUM_RUNS;
        double scalar_std = std::sqrt(scalar_var);

        std::vector<double> simd_times;
        double simd_total = 0.0;
        fmatrix output_simd(0, 0, 0, 0);
        for (int run = 0; run < NUM_RUNS; run++)
        {
            auto tbegin = steady_clock::now();
            output_simd = evaluate_simd(x, y, coef, intercept);
            auto tend = steady_clock::now();

            double time = duration_cast<microseconds>(tend - tbegin).count() / 1000.0;
            simd_times.push_back(time);
            simd_total += time;
        }
        double simd_avg = simd_total / NUM_RUNS;
        double simd_var = 0.0;
        for (double t : simd_times)
        {
            simd_var += (t - simd_avg) * (t - simd_avg);
        }
        simd_var /= NUM_RUNS;
        double simd_std = std::sqrt(simd_var);

        std::cout << "Scalar avg (" << NUM_RUNS << " runs): " << scalar_avg 
                  << "ms (std: " << scalar_std << ")" << std::endl;
        std::cout << "SIMD   avg (" << NUM_RUNS << " runs): " << simd_avg  
                  << "ms (std: " << simd_std << ")" << std::endl;
        std::cout << "Speedup:    " << (scalar_avg / simd_avg) << "x" << std::endl;
    }

    return 0;
}