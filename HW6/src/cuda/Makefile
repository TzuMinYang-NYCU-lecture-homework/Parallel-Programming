NVCC = nvcc
CXX=g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm -lOpenCL
CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g -fPIC

APP_NAME=Conv
OBJDIR=objs
COMMONDIR=./common

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'  -lOpenCL
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 -lOpenCL

PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME) output_cuda.bmp

OBJS=$(OBJDIR)/main.o $(OBJDIR)/Conv.o bmpfuncs.o helper.o

$(APP_NAME): dirs $(OBJS)
		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS)

$(OBJDIR)/main.o: main.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(COMMONDIR)/CycleTimer.h Conv.h

$(OBJDIR)/Conv.o : Conv.cu Conv.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c Conv.cu -o $@

%.o: %.c
	$(CXX) -c $(FLAGS) $< -o $@