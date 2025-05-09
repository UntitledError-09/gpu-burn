#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <sys/time.h>
#include <signal.h>
#include <fstream>
#include <cstring>
#include <cstdio>

class GPU_Test {
public:
    GPU_Test(int dev, bool doubles, bool tensors, const char *kernelFile);
    ~GPU_Test();

    static void termHandler(int signum);

    unsigned long long int getErrors();
    size_t getIters();
    void bind();
    size_t totalMemory();
    size_t availMemory();
    void initBuffers(float *A, float *B, ssize_t useBytes = 0);
    void compute();
    void initCompareKernel();
    void compare();
    bool shouldRun();

private:
    bool d_doubles;
    bool d_tensors;
    int d_devNumber;
    const char *d_kernelFile;
    size_t d_iters;
    size_t d_resultSize;

    long long int d_error;

    static const int g_blockSize = 16;

    CUdevice d_dev;
    CUcontext d_ctx;
    CUmodule d_module;
    CUfunction d_function;

    CUdeviceptr d_Cdata;
    CUdeviceptr d_Adata;
    CUdeviceptr d_Bdata;
    CUdeviceptr d_faultyElemData;
    int *d_faultyElemsHost;

    cublasHandle_t d_cublas;
};

void _checkError(int rCode, std::string file, int line, std::string desc = "");
void _checkError(cublasStatus_t rCode, std::string file, int line, std::string desc = "");
double getTime();

int initCuda();
template <class T>
void startBurn(int index, int writeFd, T *A, T *B, bool doubles, bool tensors,
               ssize_t useBytes, const char *kernelFile);
int pollTemp(pid_t *p)
void updateTemps(int handle, std::vector<int> *temps)
void listenClients(std::vector<int> clientFd, std::vector<pid_t> clientPid,
                   int runTime, std::chrono::seconds sigterm_timeout_threshold_secs);
template <class T>
void launch(int runLength, bool useDoubles, bool useTensorCores,
            ssize_t useBytes, int device_id, const char * kernelFile,
            std::chrono::seconds sigterm_timeout_threshold_secs);
void showHelp();
ssize_t decodeUSEMEM(const char *s);
int gpuburn(int argc, char **argv);
