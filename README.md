# Big Integer Multiplication

This repository implements big integer multiplication via the Number Theoretic Transform (NTT), focusing on performance on the ARM NEON architecture.
The code has been tested and benchmarked on a Raspberry Pi 4 (4-core CPU, 2 GiB RAM, 64-bit OS).

Benchmarking is performed using [aarch64-bench](https://github.com/mkannwischer/aarch64-bench).

```usage
make CYCLES=PERF
./bench
```
