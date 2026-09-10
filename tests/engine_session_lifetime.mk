# Root-owned static template. Compile only, in a fresh owned build directory.
# Darwin recipe matches the existing constructor-handoff CPU flags.
CC := /usr/bin/cc
CFLAGS := -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99
LDLIBS := -lm -pthread
OBJECTS := ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_laguna_stream.o ds4_runtime.o ds4_qualification_control.o ds4_plan_io.o ds4_laguna_plan.o ds4_layer_pack.o
.PHONY: all
all: fixture
fixture.o: ../source/tests/test_engine_session_lifetime.c ../source/ds4.c ../source/ds4_distributed.c ../source/ds4_tp.c ../source/ds4_ssd.c ../source/ds4_laguna_stream.c ../source/ds4_runtime.c ../source/ds4_qualification_control.c ../source/ds4_plan_io.c ../source/ds4_laguna_plan.c ../source/ds4_layer_pack.c ../source/ds4.h ../source/ds4_distributed.h ../source/ds4_laguna_plan.h ../source/ds4_laguna_stream.h ../source/ds4_tp.h ../source/ds4_layer_pack.h ../source/ds4_gpu_mgpu.h ../source/ds4_gpu.h ../source/ds4_ssd.h ../source/ds4_runtime.h ../source/ds4_plan_io.h ../source/ds4_gpu_resident.h ../source/ds4_streaming_hotlist.inc ../source/ds4_streaming_hotlist_glm52.inc
	$(CC) $(CFLAGS) -Wno-unused-function -I../source -c -o $@ $<
%.o: ../source/%.c
	$(CC) $(CFLAGS) -I../source -c -o $@ $<
fixture: fixture.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
