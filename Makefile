# SPDX-License-Identifier: Apache-2.0

.PHONY: clean all

CC = gcc
CXX = g++

CFLAGS := \
	-Wall \
	-Wextra \
	-Werror=unused-result \
	-Wpedantic \
	-Werror \
	-Wmissing-prototypes \
	-Wshadow \
	-Wpointer-arith \
	-Wredundant-decls \
	-Wno-long-long \
	-Wno-unknown-pragmas \
	-Wno-unused-command-line-argument \
	-O3 \
	-fomit-frame-pointer \
	-std=c99 \
	-pedantic \
	-Ihal \
	-MMD \
	$(CFLAGS)

LDFLAGS := \
	   -lgmp \
	   $(LDFLAGS)

ifeq ($(CYCLES),PMU)
	CFLAGS += -DPMU_CYCLES
endif

ifeq ($(CYCLES),PERF)
	CFLAGS += -DPERF_CYCLES
endif

ifeq ($(CYCLES),MAC)
	CFLAGS += -DMAC_CYCLES
endif

COMMON_CPP_SRCS = bigint.cpp bigint.h mul_ntt.hpp
COMMON_CPP_OBJS = $(COMMON_CPP_SRCS:.cpp=.o)

TEST_C_SRCS = test.c
TEST_OBJS = $(TEST_C_SRCS:.c=.o) $(COMMON_CPP_OBJS)

BENCH_C_SRCS = bench.c hal/hal.c
BENCH_OBJS = $(BENCH_C_SRCS:.c=.o) $(COMMON_CPP_OBJS)

TARGETS = test bench

all: $(TARGETS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TEST_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

bench: $(BENCH_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o *.d $(TARGETS)
