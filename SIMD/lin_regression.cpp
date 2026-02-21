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
    //
    // You CANNOT use threads.
    // We are forbidding multithreading to make the coding take less time and
    // get you to focus on learning the SIMD part.

    return output;
}

int main(int argc, char *argv[])
{
    // These are four linear regression models
    auto &&[x, y, coef, intercept] = read_bin_data("data/calhouse.bin");
    //auto &&[x, y, coef, intercept] = read_bin_data("data/allstate.bin");
    //auto &&[x, y, coef, intercept] = read_bin_data("data/diamonds.bin");
    //auto &&[x, y, coef, intercept] = read_bin_data("data/cpusmall.bin");

    // This is a logistic regression model, but can be evaluated in the same way
    // All you would need to do is apply the sigmoid to the values in `output_*`
    //auto &&[x, y, coef, intercept] = read_bin_data("data/mnist_5vall.bin");
    
    // TODO repeat the number of time measurements to get a more accurate
    // estimate of the runtime.

    steady_clock::time_point tbegin, tend;

    // SCALAR
    tbegin = steady_clock::now();
    auto output_scalar = evaluate_scalar(x, y, coef, intercept);
    tend = steady_clock::now();

    std::cout << "Evaluated scalar in "
        << (duration_cast<microseconds>(tend-tbegin).count()/1000.0)
        << "ms" << std::endl;

    // SIMD
    tbegin = steady_clock::now();
    auto output_simd = evaluate_simd(x, y, coef, intercept);
    tend = steady_clock::now();

    std::cout << "Evaluated SIMD in "
        << (duration_cast<microseconds>(tend-tbegin).count()/1000.0)
        << "ms" << std::endl;

    // TODO check output
}
