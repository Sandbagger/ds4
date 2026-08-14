CC ?= cc
CXX ?= c++
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
SAMPLING_TEST :=
TEST_GC_LDFLAGS := -Wl,-dead_strip
else
NATIVE_CPU_FLAG ?= -march=native
SAMPLING_TEST := tests/test_sampling
TEST_GC_LDFLAGS := -Wl,--gc-sections
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
CXXFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c++11
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support.gguf
DS4_BUILD_REVISION := $(shell git rev-parse --verify HEAD 2>/dev/null)
DS4_BUILD_DIRTY := $(if $(shell git status --porcelain --untracked-files=no 2>/dev/null),1,0)
DS4_BUILD_INFO_OBJ ?= ds4_build_info.o
CPU_BUILD_INFO_OBJ := ds4_build_info_cpu.o

ifeq ($(UNAME_S),Darwin)
DS4_BUILD_DEFAULT_BACKEND := metal
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_metal.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_layer_pack.o
else
DS4_BUILD_DEFAULT_BACKEND := cuda
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_cuda.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_layer_pack.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -ldl
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test test-cuda-build-contract test-laguna-compact-python test-laguna-compact-contract test-laguna-runtime-identity test-laguna-server-contract test-metal-session-batch test-session-logits-only-policy test-session-request-attribution-api test-laguna-stream test-laguna-plan test-runtime test-runtime-request test-qualification-control test-cuda-session-batch test-cuda-mixed-batch test-cuda-laguna-kernels test-cuda-laguna-model test-cuda-laguna-stream test-cuda-laguna-request-counters test-cuda-laguna-model-page-advice test-cuda-laguna-external-attribution test-cuda-laguna-qualification-control test-cuda-laguna-runtime-identity test-cuda-laguna-resident test-cuda-laguna-streaming test-cuda-laguna-c7 dspark-acceptance dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression strix-halo rocm FORCE_BUILD_INFO

tests/test_session_request_attribution_api.o: tests/test_session_request_attribution_api.c ds4.h ds4_runtime.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_session_request_attribution_api: tests/test_session_request_attribution_api.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-session-request-attribution-api: tests/test_session_request_attribution_api
	./tests/test_session_request_attribution_api

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) -I. -c -o $@ $<

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ) $(METAL_LDLIBS)

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o
	$(CC) $(QUALITY_CFLAGS) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(CPU_BUILD_INFO_OBJ)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make test-cuda-laguna-resident  Run the pinned Poolside resident-CUDA oracle"
	@echo "  make test-cuda-laguna-streaming DS4_TEST_MODEL=/abs/model.gguf DS4_QUALIFICATION_PLAN=/abs/plan.json DS4_QUALIFICATION_PLAN_SHA256=<sha256>  Run the descriptor-bound streamed CUDA gate"
	@echo "  make test-cuda-laguna-model-page-advice DS4_TEST_MODEL=/abs/model.gguf  Run exact-inode page-disposal qualification"
	@echo "  make test-cuda-laguna-request-counters DS4_TEST_MODEL=/abs/model.gguf  Run two-session request-counter attribution"
	@echo "  make test-cuda-laguna-external-attribution DS4_TEST_MODEL=/abs/model.gguf  Reconcile live smaps/mincore/NVML footprint"
	@echo "  make test-cuda-laguna-qualification-control DS4_TEST_MODEL=/abs/model.gguf  Run live descriptor/barrier fail-closed qualification"
	@echo "  make test-cuda-laguna-runtime-identity DS4_TEST_MODEL=/abs/model.gguf  Run the complete Task 16 host + live CUDA gate"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o" \
		CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_BUILD_INFO_OBJ="ds4_build_info_rocm.o" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(DS4_BUILD_INFO_OBJ)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(CPU_BUILD_INFO_OBJ)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) ds4_build_info_cpu.o $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke tests/test_cuda_laguna_kernels tests/test_cuda_laguna_stream
	./tests/cuda_long_context_smoke
	env -u DS4_CUDA_MOE_DECODE_GRAPH ./tests/test_cuda_laguna_kernels --case all
	./tests/test_cuda_laguna_stream --case nvml-fd-stability
	./tests/test_cuda_laguna_stream --case startup
endif

ifeq ($(UNAME_S),Darwin)
ds4_build_info.o: ds4_build_info.c ds4_build_info.h ds4_runtime.h FORCE_BUILD_INFO
	$(CC) $(CFLAGS) -DDS4_BUILD_REVISION=\"$(DS4_BUILD_REVISION)\" -DDS4_BUILD_DIRTY=$(DS4_BUILD_DIRTY) -DDS4_BUILD_BACKEND=\"metal\" -DDS4_BUILD_FEATURES=\"laguna,ssd_streaming\" -c -o $@ $<
else
ds4_build_info.o: ds4_build_info.c ds4_build_info.h ds4_runtime.h FORCE_BUILD_INFO
	$(CC) $(CFLAGS) -DDS4_BUILD_REVISION=\"$(DS4_BUILD_REVISION)\" -DDS4_BUILD_DIRTY=$(DS4_BUILD_DIRTY) -DDS4_BUILD_BACKEND=\"cuda\" -DDS4_BUILD_FEATURES=\"laguna,ssd_streaming\" -c -o $@ $<
endif

ds4_build_info_cpu.o: ds4_build_info.c ds4_build_info.h ds4_runtime.h FORCE_BUILD_INFO
	$(CC) $(CFLAGS) -DDS4_BUILD_REVISION=\"$(DS4_BUILD_REVISION)\" -DDS4_BUILD_DIRTY=$(DS4_BUILD_DIRTY) -DDS4_BUILD_BACKEND=\"cpu\" -DDS4_BUILD_FEATURES=\"laguna,ssd_streaming\" -c -o $@ $<

ds4_build_info_rocm.o: ds4_build_info.c ds4_build_info.h ds4_runtime.h FORCE_BUILD_INFO
	$(CC) $(CFLAGS) -DDS4_BUILD_REVISION=\"$(DS4_BUILD_REVISION)\" -DDS4_BUILD_DIRTY=$(DS4_BUILD_DIRTY) -DDS4_BUILD_BACKEND=\"rocm\" -DDS4_BUILD_FEATURES=\"laguna,ssd_streaming\" -c -o $@ $<

FORCE_BUILD_INFO:

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_laguna_stream.h ds4_laguna_plan.h ds4_plan_io.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h ds4_gpu.h ds4_laguna_plan.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h ds4_laguna_plan.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_laguna_stream.h ds4_laguna_plan.h ds4_plan_io.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_build_info.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h ds4_laguna_plan.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_laguna_plan.h ds4_laguna_stream.h ds4_runtime.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_laguna_plan.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

ds4_rocm_compat.o: ds4_rocm_compat.cu ds4_gpu.h ds4_laguna_plan.h ds4_gpu_mgpu.h ds4_gpu_args.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_compat.cu

ds4_rocm_unavailable.o: ds4_rocm_unavailable.cu
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_unavailable.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o ds4_laguna_stream.o ds4_runtime.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_laguna_kernels.o: tests/test_cuda_laguna_kernels.c ds4_gpu.h ds4_laguna_plan.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/ds4_cuda_laguna_kernels_test_hooks.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_laguna_plan.h ds4_laguna_stream.h ds4_runtime.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -DDS4_TEST_HOOKS -c -o $@ ds4_cuda.cu

tests/test_cuda_laguna_kernels: tests/test_cuda_laguna_kernels.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_laguna_stream.o ds4_runtime.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-laguna-kernels: tests/test_cuda_laguna_kernels
	env -u DS4_CUDA_MOE_DECODE_GRAPH ./tests/test_cuda_laguna_kernels --case all

tests/test_cuda_q4k_mmvq_microscope.o: tests/test_cuda_q4k_mmvq_microscope.c ds4_gpu.h ds4_laguna_plan.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_q4k_mmvq_microscope: tests/test_cuda_q4k_mmvq_microscope.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_laguna_stream.o ds4_runtime.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-q4k-mmvq-microscope: tests/test_cuda_q4k_mmvq_microscope
	./tests/test_cuda_q4k_mmvq_microscope

tests/test_cuda_f32_mmvf_microscope.o: tests/test_cuda_f32_mmvf_microscope.c ds4_gpu.h ds4_laguna_plan.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_f32_mmvf_microscope: tests/test_cuda_f32_mmvf_microscope.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_laguna_stream.o ds4_runtime.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-f32-mmvf-microscope: tests/test_cuda_f32_mmvf_microscope
	./tests/test_cuda_f32_mmvf_microscope

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_laguna_stream.o: tests/test_laguna_stream.c ds4.h ds4_ssd.h ds4_laguna_stream.h ds4_runtime.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

ds4_laguna_stream.o: ds4_laguna_stream.c ds4_laguna_stream.h ds4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ ds4_laguna_stream.c

ds4_runtime.o: ds4_runtime.c ds4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ ds4_runtime.c

ds4_qualification_control.o: ds4_qualification_control.c ds4_runtime.h
	$(CC) $(CFLAGS) -c -o $@ ds4_qualification_control.c

tests/test_runtime.o: tests/test_runtime.c ds4_runtime.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_runtime: tests/test_runtime.o ds4_runtime.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-runtime: tests/test_runtime
	./tests/test_runtime --case external-attribution
	./tests/test_runtime --case request-metrics

test-runtime-request: tests/test_runtime
	./tests/test_runtime --case request-metrics

test-laguna-server-contract: ds4_test
	python3 tests/test_laguna_server_contract.py --server ./ds4_test --case admission --case metrics -v
	python3 tests/test_laguna_server_live_contract.py -v

tests/test_qualification_control.o: tests/test_qualification_control.c ds4.h ds4_runtime.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_qualification_control: tests/test_qualification_control.o ds4_qualification_control.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-qualification-control: tests/test_qualification_control
	./tests/test_qualification_control
	python3 tests/test_qualification_control_contract.py -v
	python3 tests/test_qualification_control_cli_contract.py -v

ds4_plan_io.o: ds4_plan_io.c ds4_plan_io.h
	$(CC) $(CFLAGS) -c -o $@ ds4_plan_io.c

ds4_plan_io_test_hooks.o: ds4_plan_io.c ds4_plan_io.h
	$(CC) $(CFLAGS) -DDS4_PLAN_IO_TEST_HOOKS -c -o $@ ds4_plan_io.c

ds4_laguna_plan.o: ds4_laguna_plan.c ds4_laguna_plan.h ds4_laguna_stream.h ds4_runtime.h ds4_plan_io.h
	$(CC) $(CFLAGS) -c -o $@ ds4_laguna_plan.c

ds4_bound_test_hooks.o: ds4.c ds4.h ds4_ssd.h ds4_laguna_stream.h ds4_laguna_plan.h ds4_plan_io.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS \
		-DDS4_TEST_FORCE_GRAPH_CACHE_F32 -ffunction-sections \
		-fdata-sections -c -o $@ ds4.c

tests/test_laguna_stream: tests/test_laguna_stream.o ds4_bound_test_hooks.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o
	$(CC) $(CFLAGS) $(TEST_GC_LDFLAGS) -o $@ $^ $(LDLIBS)

tests/test_plan_io.o: tests/test_plan_io.c ds4_plan_io.h
	$(CC) $(CFLAGS) -DDS4_PLAN_IO_TEST_HOOKS -I. -c -o $@ $<

tests/test_plan_io: tests/test_plan_io.o ds4_plan_io_test_hooks.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_laguna_plan.o: tests/test_laguna_plan.c ds4_laguna_plan.h ds4_laguna_stream.h ds4_runtime.h ds4_plan_io.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_laguna_plan: tests/test_laguna_plan.o ds4_laguna_plan.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-laguna-plan: tests/test_plan_io tests/test_laguna_plan
	./tests/test_plan_io
	./tests/test_laguna_plan

tests/test_runtime_cpp_link.o: tests/test_runtime_cpp_link.cc ds4.h ds4_gpu.h ds4_laguna_plan.h ds4_laguna_stream.h ds4_runtime.h
	$(CXX) $(CXXFLAGS) -I. -c -o $@ $<

tests/test_runtime_cpp_link: tests/test_runtime_cpp_link.o ds4_laguna_stream.o ds4_runtime.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

test-laguna-stream: tests/test_laguna_stream tests/test_runtime_cpp_link
	./tests/test_laguna_stream --case options
	./tests/test_laguna_stream --case ledger
	./tests/test_laguna_stream --case allocation
	./tests/test_laguna_stream --case cache-policy
	./tests/test_laguna_stream --case grouping
	./tests/test_laguna_stream --case prefill-plan
	./tests/test_laguna_stream --case page-ranges
	./tests/test_runtime_cpp_link

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_laguna_stream.h ds4_laguna_plan.h ds4_plan_io.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_logits_only.o: tests/test_session_logits_only.c ds4.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_session_logits_only: tests/test_session_logits_only.o ds4_cpu_test_hooks.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-session-logits-only-policy: tests/test_session_logits_only
	./tests/test_session_logits_only

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_laguna_plan.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_cuda.o ds4_laguna_stream.o ds4_runtime.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h ds4_laguna_plan.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_cuda.o ds4_laguna_stream.o ds4_runtime.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_laguna_plan.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_cuda.o ds4_laguna_stream.o ds4_runtime.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_laguna_stream.h ds4_laguna_plan.h ds4_plan_io.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch

tests/test_cuda_laguna_model.o: tests/test_cuda_laguna_model.c ds4.h ds4_gpu.h ds4_gpu_args.h ds4_laguna_plan.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_laguna_model: tests/test_cuda_laguna_model.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/probe_ds4_laguna_moe.o: tests/oracle-producers/laguna-c7/probe_ds4_laguna_moe.c ds4.h ds4_gpu_args.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/probe_ds4_laguna_moe: tests/probe_ds4_laguna_moe.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o tests/ds4_cuda_laguna_kernels_test_hooks.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/probe_ds4_laguna_moe_release.o: tests/oracle-producers/laguna-c7/probe_ds4_laguna_moe.c ds4.h ds4_gpu_args.h
	$(CC) $(CFLAGS) -DDS4_LAGUNA_RELEASE_CONTROL -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/probe_ds4_laguna_moe_release: tests/probe_ds4_laguna_moe_release.o ds4.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o ds4_cuda.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/probe_ds4_laguna_behavior.o: tests/oracle-producers/laguna-c7/probe_ds4_laguna_behavior.c ds4.h ds4_gpu_args.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/probe_ds4_laguna_behavior: tests/probe_ds4_laguna_behavior.o ds4.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_plan_io.o ds4_laguna_plan.o ds4_cuda.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-laguna-model: tests/test_cuda_laguna_model
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_laguna_model --mode streamed --case short
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_laguna_model --mode streamed --case prefill-8192
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_laguna_model --mode resident --case all

tests/test_cuda_laguna_stream.o: tests/test_cuda_laguna_stream.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_laguna_plan.h ds4_laguna_stream.h ds4_plan_io.h ds4_runtime.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/ds4_cuda_laguna_stream_test_hooks.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_laguna_plan.h ds4_laguna_stream.h ds4_runtime.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -DDS4_TEST_HOOKS -c -o $@ ds4_cuda.cu

tests/test_cuda_laguna_stream: tests/test_cuda_laguna_stream.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o tests/ds4_cuda_laguna_stream_test_hooks.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-laguna-stream: tests/test_cuda_laguna_stream
	timeout 60s ./tests/test_cuda_laguna_stream --case nvml-fd-stability
	timeout 60s ./tests/test_cuda_laguna_stream --case startup
	timeout 60s ./tests/test_cuda_laguna_stream --case cache-validation
	timeout 60s ./tests/test_cuda_laguna_stream --case cache-io
	timeout 60s ./tests/test_cuda_laguna_stream --case cache-faults
	timeout 60s ./tests/test_cuda_laguna_stream --case cache-unsafe
	timeout 60s ./tests/test_cuda_laguna_stream --case cache-unsafe-race
	timeout 60s ./tests/test_cuda_laguna_stream --case prefill-allocation
	timeout 60s ./tests/test_cuda_laguna_stream --case page-advice
	timeout 60s ./tests/test_cuda_laguna_stream --case create-unwind-unsafe
	timeout 60s ./tests/test_cuda_laguna_stream --case teardown-unsafe

# Model-bearing qualification is intentionally separate from the synthetic
# stream suite: it cold-prepares and scans the exact GGUF inode and can take
# several minutes on a 68 GB Laguna model.
test-cuda-laguna-model-page-advice: tests/test_cuda_laguna_stream
	@if [ "$(DS4_TEST_MODEL)" = ds4flash.gguf ]; then \
		echo "error: set DS4_TEST_MODEL to the explicit Laguna GGUF path" >&2; \
		exit 2; \
	fi
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" timeout --kill-after=5s 900s \
		./tests/test_cuda_laguna_stream --case model-page-advice

# Run in a fresh process so the process-level physical baseline contains only
# the two explicitly attributed sessions created by this case.
test-cuda-laguna-request-counters: tests/test_cuda_laguna_stream
	@if [ "$(DS4_TEST_MODEL)" = ds4flash.gguf ]; then \
		echo "error: set DS4_TEST_MODEL to the explicit Laguna GGUF path" >&2; \
		exit 2; \
	fi
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" timeout --kill-after=5s 900s \
		./tests/test_cuda_laguna_stream --case request-counters

# Capture the frozen peer inventory before the test process makes any CUDA
# call.  Descriptor-bound safe-union cold preparation remains the Python
# qualification harness's responsibility; this target measures live state.
test-cuda-laguna-external-attribution: tests/test_cuda_laguna_stream
	@if [ "$(DS4_TEST_MODEL)" = ds4flash.gguf ]; then \
		echo "error: set DS4_TEST_MODEL to the explicit Laguna GGUF path" >&2; \
		exit 2; \
	fi
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" timeout --kill-after=5s 900s \
		./tests/test_cuda_laguna_stream --case external-attribution

# Run success and peer-loss as separate processes: an unsafe attribution
# latch is intentionally process lifetime state and must not contaminate the
# successful transaction's evidence or teardown assertions.
test-cuda-laguna-qualification-control: tests/test_cuda_laguna_stream
	@if [ "$(origin DS4_LOCK_FILE)" != undefined ]; then \
		echo "error: DS4_LOCK_FILE is forbidden in test-cuda-laguna-qualification-control" >&2; \
		exit 2; \
	fi
	@if [ -z "$${DS4_TEST_MODEL:-}" ] || [ "$${DS4_TEST_MODEL}" = ds4flash.gguf ]; then \
		echo "error: set DS4_TEST_MODEL to the explicit Laguna GGUF path" >&2; \
		exit 2; \
	fi
	DS4_TEST_MODEL="$${DS4_TEST_MODEL}" timeout --kill-after=5s 900s \
		./tests/test_cuda_laguna_stream --case qualification-control-success
	DS4_TEST_MODEL="$${DS4_TEST_MODEL}" timeout --kill-after=5s 900s \
		./tests/test_cuda_laguna_stream --case qualification-control-disconnect
test-cuda-laguna-qualification-control: override DS4_TEST_MODEL := $(value DS4_TEST_MODEL)

test-cuda-laguna-runtime-identity: test-laguna-runtime-identity test-cuda-laguna-qualification-control

export DS4_TEST_MODEL
export LAGUNA_TOKENIZER_RUNTIME_COMMIT
export DS4_QUALIFICATION_PLAN
export DS4_QUALIFICATION_PLAN_SHA256

test-cuda-laguna-resident: tests/test_cuda_laguna_kernels tests/test_cuda_laguna_model
	tests/run_cuda_laguna_gate.sh resident
test-cuda-laguna-resident: override DS4_TEST_MODEL := $(value DS4_TEST_MODEL)
test-cuda-laguna-resident: override LAGUNA_TOKENIZER_RUNTIME_COMMIT := $(value LAGUNA_TOKENIZER_RUNTIME_COMMIT)

test-cuda-laguna-streaming: test-laguna-stream tests/test_cuda_laguna_model tests/test_cuda_laguna_stream
	tests/run_cuda_laguna_gate.sh streaming
test-cuda-laguna-streaming: override DS4_TEST_MODEL := $(value DS4_TEST_MODEL)
test-cuda-laguna-streaming: override LAGUNA_TOKENIZER_RUNTIME_COMMIT := $(value LAGUNA_TOKENIZER_RUNTIME_COMMIT)
test-cuda-laguna-streaming: override DS4_QUALIFICATION_PLAN := $(value DS4_QUALIFICATION_PLAN)
test-cuda-laguna-streaming: override DS4_QUALIFICATION_PLAN_SHA256 := $(value DS4_QUALIFICATION_PLAN_SHA256)

test-cuda-laguna-c7: tests/test_cuda_laguna_kernels tests/test_cuda_laguna_model tests/test_cuda_laguna_stream
	tests/run_cuda_laguna_gate.sh c7
test-cuda-laguna-c7: override DS4_TEST_MODEL := $(value DS4_TEST_MODEL)
test-cuda-laguna-c7: override LAGUNA_TOKENIZER_RUNTIME_COMMIT := $(value LAGUNA_TOKENIZER_RUNTIME_COMMIT)
endif

ifeq ($(UNAME_S),Darwin)
test-cuda-laguna-c7:
	@echo "error: test-cuda-laguna-c7 is unsupported; requires CUDA on Linux" >&2; exit 2

test-cuda-laguna-streaming:
	@echo "error: test-cuda-laguna-streaming is unsupported; requires CUDA on Linux" >&2; exit 2

test-cuda-laguna-model-page-advice:
	@echo "error: test-cuda-laguna-model-page-advice is unsupported; requires CUDA on Linux" >&2; exit 2

test-cuda-laguna-request-counters:
	@echo "error: test-cuda-laguna-request-counters is unsupported; requires CUDA on Linux" >&2; exit 2

test-cuda-laguna-external-attribution:
	@echo "error: test-cuda-laguna-external-attribution is unsupported; requires CUDA on Linux" >&2; exit 2

test-cuda-laguna-qualification-control:
	@echo "error: test-cuda-laguna-qualification-control is unsupported; requires CUDA on Linux" >&2; exit 2

test-cuda-laguna-runtime-identity:
	@echo "error: test-cuda-laguna-runtime-identity is unsupported; requires CUDA on Linux" >&2; exit 2
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test-cuda-build-contract:
	python3 tests/test_cuda_build_contract.py -v
	python3 tests/test_main_session_recovery_hardening_contract.py -v

test-laguna-compact-python:
	@command -v uv >/dev/null 2>&1 || { \
		echo "error: test-laguna-compact-python requires uv" >&2; \
		exit 127; \
	}
	@uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python -c 'from jsonschema import Draft202012Validator' || { \
		echo "error: unable to provision pinned jsonschema==4.25.1 with uv" >&2; \
		exit 1; \
	}
	uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python gguf-tools/quality-testing/test_compact_runtime_qualify.py -v

test-laguna-compact-contract:
	@command -v uv >/dev/null 2>&1 || { \
		echo "error: test-laguna-compact-contract requires uv" >&2; \
		exit 127; \
	}
	@uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python -c 'from jsonschema import Draft202012Validator; import rfc3339_validator, rfc8785' || { \
		echo "error: unable to provision pinned compact-runtime requirements with uv" >&2; \
		exit 1; \
	}
	uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python tests/test_runtime_contract.py -v

test-laguna-runtime-identity: tests/test_runtime tests/test_qualification_control ds4 ds4-server ds4-agent ds4-bench ds4-eval
	@command -v uv >/dev/null 2>&1 || { \
		echo "error: test-laguna-runtime-identity requires uv" >&2; \
		exit 127; \
	}
	@uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python -c 'from jsonschema import Draft202012Validator; import rfc3339_validator, rfc8785' || { \
		echo "error: unable to provision pinned compact-runtime requirements with uv" >&2; \
		exit 1; \
	}
	python3 tests/test_task16_gate_contract.py -v
	./tests/test_runtime --case external-attribution
	./tests/test_qualification_control
	python3 tests/test_qualification_control_contract.py -v
	python3 tests/test_qualification_control_cli_contract.py -v
	uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python gguf-tools/quality-testing/test_compact_runtime_qualify.py -v
	uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python tests/test_version_json.py -v
	uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python tests/test_runtime_contract.py -v
	DS4_RUNTIME_SERVER_URL= uv run --with-requirements gguf-tools/quality-testing/requirements-compact-runtime.txt \
		python tests/test_runtime_endpoint_contract.py -v

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test test-cuda-build-contract test-laguna-compact-python test-laguna-server-contract \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	tests/test_session_logits_only tests/test_laguna_stream tests/test_runtime tests/test_runtime_cpp_link \
	tests/test_plan_io tests/test_laguna_plan $(SAMPLING_TEST) ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_session_logits_only
	./tests/test_laguna_stream --case options
	./tests/test_laguna_stream --case ledger
	./tests/test_laguna_stream --case allocation
	./tests/test_laguna_stream --case cache-policy
	./tests/test_laguna_stream --case grouping
	./tests/test_laguna_stream --case prefill-plan
	./tests/test_laguna_stream --case page-ranges
	./tests/test_runtime --case external-attribution
	./tests/test_runtime --case request-metrics
	./tests/test_runtime_cpp_link
	./tests/test_plan_io
	./tests/test_laguna_plan
	./tests/test_gpu_args_cli.sh
ifneq ($(UNAME_S),Darwin)
	./tests/test_sampling
endif

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run make dspark-support or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: run ./download_model.sh mtp or set DS4_TEST_MTP=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o tests/test_q4k_dot tests/test_metal_session_batch tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_session_logits_only tests/test_session_logits_only.o tests/test_laguna_stream tests/test_runtime tests/test_runtime_cpp_link tests/test_qualification_control tests/test_plan_io tests/test_laguna_plan tests/test_laguna_stream.o tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/test_cuda_laguna_kernels tests/test_cuda_q4k_mmvq_microscope tests/test_cuda_f32_mmvf_microscope tests/test_cuda_laguna_model tests/test_cuda_laguna_model.o tests/probe_ds4_laguna_moe tests/probe_ds4_laguna_moe.o tests/probe_ds4_laguna_moe_release tests/probe_ds4_laguna_moe_release.o tests/probe_ds4_laguna_behavior tests/probe_ds4_laguna_behavior.o tests/test_cuda_laguna_stream tests/test_cuda_laguna_stream.o tests/*.o *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
