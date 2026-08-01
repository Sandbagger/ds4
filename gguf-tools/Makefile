CC ?= cc
CXX ?= c++
UNAME_S := $(shell uname -s)
LLAMA_CPP_DIR ?= ../../deepseek-v4-quants/llama.cpp
LLAMA_CPP_BUILD ?= $(LLAMA_CPP_DIR)/build-remote
LLAMA_CPP_BIN ?= $(LLAMA_CPP_BUILD)/bin
override LLAMA_LAGUNA_COMMIT := 04b2b72cb54048ead292884adbe11f284e3ec950
LLAMA_CPP_CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 \
	-I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal
QUALITY_TARGETS := ds4.o ds4_metal.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o rax.o
QUALITY_OBJS := ../ds4.o ../ds4_metal.o ../ds4_distributed.o ../ds4_tp.o ../ds4_ssd.o ../ds4_layer_pack.o ../rax.o
QUALITY_LINK := $(CC) $(CFLAGS) -I.. -o
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -lggml-blas -lggml-metal \
	-framework Foundation -framework Metal -framework Accelerate
else
NATIVE_CPU_FLAG ?= -march=native
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
QUALITY_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
QUALITY_TARGETS := ds4.o ds4_cuda.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o rax.o
QUALITY_OBJS := ../ds4.o ../ds4_cuda.o ../ds4_distributed.o ../ds4_tp.o ../ds4_ssd.o ../ds4_layer_pack.o ../rax.o
QUALITY_LINK := $(NVCC) $(NVCCFLAGS) -I.. -o
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -ldl -lpthread
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quality-score quality-llama-score quality-laguna-logits

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)

quality-llama-score:
	$(CXX) $(LLAMA_CPP_CXXFLAGS) -o quality-testing/score_llama quality-testing/score_llama.cpp $(LLAMA_CPP_LDLIBS)

quality-laguna-logits:
	@actual_commit="$$(git -C "$(LLAMA_CPP_DIR)" rev-parse HEAD 2>/dev/null)" || { \
		echo "error: cannot read llama.cpp HEAD from $(LLAMA_CPP_DIR)" >&2; \
		exit 1; \
	}; \
	if [ "$$actual_commit" != "$(LLAMA_LAGUNA_COMMIT)" ]; then \
		echo "error: Laguna oracle requires llama.cpp HEAD $(LLAMA_LAGUNA_COMMIT), got $$actual_commit" >&2; \
		exit 1; \
	fi; \
	dirty="$$(git -C "$(LLAMA_CPP_DIR)" status --porcelain --untracked-files=no 2>/dev/null)" || { \
		echo "error: cannot inspect llama.cpp checkout $(LLAMA_CPP_DIR)" >&2; \
		exit 1; \
	}; \
	if [ -n "$$dirty" ]; then \
		echo "error: llama.cpp tracked checkout is dirty" >&2; \
		printf '%s\n' "$$dirty" >&2; \
		exit 1; \
	fi
	$(CXX) $(LLAMA_CPP_CXXFLAGS) -o quality-testing/dump_llama_logits quality-testing/dump_llama_logits.cpp $(LLAMA_CPP_LDLIBS)

clean:
	rm -f deepseek4-quantize quality-testing/score_official quality-testing/score_llama quality-testing/dump_llama_logits
