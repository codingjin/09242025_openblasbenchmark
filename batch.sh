#!/bin/bash

./settings.sh

make

./openblas_sgemm 4095 127 4095

./openblas_sgemm 4096 128 4096

./openblas_sgemm 4097 129 4097


./openblas_sgemm 127 8191 4095

./openblas_sgemm 128 8192 4096

./openblas_sgemm 129 8193 4097


./openblas_sgemm 127 4095 8191

./openblas_sgemm 128 4096 8192

./openblas_sgemm 129 4097 8193


./openblas_sgemm 4095 4095 4095

./openblas_sgemm 4096 4096 4096

./openblas_sgemm 4097 4097 4097
