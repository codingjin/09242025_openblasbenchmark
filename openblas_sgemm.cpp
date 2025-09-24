#include <cblas.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <string>
#include <random>
#include <cmath>
#include <unordered_map>
#include <cassert>
#include <unordered_set>
#include <cstdlib>
#include <cstring>

const int WARMUP = 100;
const int RUNS = 1000;
//const int WARMUP = 10;
//const int RUNS = 10;
const float ERR = 0.05;
double FLOPs;

void matmul(const float * const A, const float * const B, float * const C, const int M, const int N, const int K);
void compare(const float * const A, const float * const B, const int M, const int N);
void initInputs(float * const A, float * const B, const int M, const int N, const int K);
void printResults(const std::vector<double>& results, const double FLOPs);
double openblas(const float *A, const float *B, float *C, const int M, const int N, const int K);


int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Invalid Usage!\n";
        std::cout << "Usage: ./openblas_sgemm M N K\n";
        std::cout << "M N K are positive Integer for problem sizes\n";
        exit(1);
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    assert(M > 0 && N > 0 && K > 0);
    FLOPs = 2.0 * M * K * N;

    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
    float *A = (float*)malloc(sizeof(float) * M * K);
    float *B = (float*)malloc(sizeof(float) * K * N);
    initInputs(A, B, M, N, K);
    float *C = (float*)malloc(sizeof(float) * M * N);
    std::memset(C, 0, sizeof(float) * M * N);
    /*
    float *D = (float*)malloc(sizeof(float) * M * N);
    std::memset(D, 0, sizeof(float) * M * N);
    */
    /*
    // check correctness first
    matmul(A, B, D, M, N, K);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1,      // Alpha
        A, K,   // A and strides between rows
        B, N,   // B and strides between rows
        0,      // Beta
        C, N    // C and strides between rows
    );
    compare(C, D, M, N);
    */

    std::vector<double> timings;
    for (int i = 0; i < WARMUP; ++i) cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
    for (int i = 0; i < RUNS; ++i) timings.push_back(openblas(A, B, C, M, K, N));
    std::sort(timings.begin(), timings.end());
    printResults(timings, FLOPs);

    // check correctness last
    //compare(C, D, M, N);
    free(A);
    free(B);
    free(C);
    //free(D);
    return 0;
}

double openblas(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

void initInputs(float * const A, float * const B, const int M, const int N, const int K)
{
    std::mt19937 engine{137};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) A[i] = dist(engine);
    for (int i = 0; i < K * N; ++i) B[i] = dist(engine);
}

void matmul(const float * const A, const float * const B, float * const C, const int M, const int N, const int K)
{
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void compare(const float * const A, const float * const B, const int M, const int N)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::abs(A[i * N + j] - B[i * N + j]) > ERR) {
                std::cerr << "Correctness check failed!\n" << "M=" << M << " N=" << N << std::endl;
                std::cerr << "i=" << i << ", j=" << j << " A=" << A[i*M+j] << ", B=" << B[i*M+j] << std::endl;
                exit(1);
            }
        }
    }
}

void printResults(const std::vector<double>& results, const double FLOPs)
{
    double total = std::accumulate(results.begin(), results.end(), 0.0);
    double avg = total/results.size();
    double dev = 0.0;
    for (const auto& re : results)
        dev += (re - avg) * (re - avg);
    dev /= results.size();
    double stddev = std::sqrt(dev);

    std::cout << "Took " << total << " seconds for " << RUNS << " runs.\t" << WARMUP << " warmups.\n";
    std::cout << "Avg time: " << avg <<" s, stddev: " << stddev << "\n";
    std::cout << "Performance: " << FLOPs/1.0e9/avg << " GFLOPs\n\n";
}