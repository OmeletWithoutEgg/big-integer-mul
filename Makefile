# SPDX-License-Identifier: Apache-2.0

.PHONY: clean all

CC = gcc
CXX = g++
# CC = /bin/aarch64-linux-gnu-gcc
# CXX = /bin/aarch64-linux-gnu-g++

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
	-std=c99 \
	-fomit-frame-pointer \
	-pedantic \
	-Ihal \
	-MMD \
	-Wfatal-errors \
	-march=native \
	$(CFLAGS)

CXXFLAGS := \
	-O3 \
	-std=c++2a \
	-pedantic \
	-Ihal \
	-MMD \
	-Wfatal-errors \
	-march=native \
	$(CXXFLAGS)

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

COMMON_CPP_SRCS = bigint.cpp
COMMON_CPP_OBJS = $(COMMON_CPP_SRCS:.cpp=.o)

BENCH_C_SRCS = bench.c hal/hal.c
BENCH_OBJS = $(BENCH_C_SRCS:.c=.o) $(COMMON_CPP_OBJS)

TEST_C_SRCS = test.c
TEST_OBJS = $(TEST_C_SRCS:.c=.o) $(COMMON_CPP_OBJS)

DEPS = $(BENCH_OBJS:.o=.d) $(TEST_OBJS:.o=.d)

TARGETS = test bench

all: $(TARGETS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

bench: $(BENCH_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o *.d $(TARGETS)

-include $(DEPS)
