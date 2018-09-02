#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>

/*
GPU #0: 'GeForce GTX 1060 6GB' @ 1708.5MHz
	Compute: 6.1
	Multi Processors(SM): 10

=== MEMORY ===
	Global Memory : 5.93457 GB
	Bus Width : 192-bit @ 4004 MHz ===> Max Bandwidth: 192 GB/s
	Constant Memory : 64 KB
	Shared Memory per Block: 48 KB
	Unified Memory Addressing: true
=== THREAD ===
	Max Threads/SM: 2048
	Threads/Block: 1024
	Max Thread Size: 1024 x 1024 x 64
	Warp Size: 32
	Max Grid Size: 2147483647 x 65535 x 65535
	Registers/SM: 64 KB
	Registers/Block: 64 KB
=== TEXTURE ===
	Texture Size 1D: 131072
	Texture Size 2D: 131072 x 65536
	Texture Size 3D: 16384 x 16384 x 16384
 */

void checkCudaError(cudaError_t error_id) {
    if (error_id != cudaSuccess) {
        std::cerr << cudaGetErrorString(error_id) << std::endl;
        exit(-1);
    }
}

int main() {
    using namespace std;
    cudaError_t errorId;

    int deviceCount = 0;
    errorId = cudaGetDeviceCount(&deviceCount);
    checkCudaError(errorId);

    for (int deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++) {
        cudaSetDevice(deviceIndex);

        cudaDeviceProp prop;
        errorId = cudaGetDeviceProperties(&prop, deviceIndex);
        checkCudaError(errorId);

        cout << "GPU #" << deviceIndex << ": '" << prop.name << "' @ " << prop.clockRate / 1000.0f << "MHz" << endl;
        cout << "\tCompute: " << prop.major << "." << prop.minor << endl;
        cout << "\tMulti Processors(SM): " << prop.multiProcessorCount << endl;
        cout << endl;
        cout << "=== MEMORY ===" << endl;
        cout << "\tGlobal Memory : " << prop.totalGlobalMem / pow(2.0f, 30.f) << " GB" << endl;
        cout << "\tBus Width : " << prop.memoryBusWidth << "-bit @ " << prop.memoryClockRate / 1000.0f << " MHz"
             << " ===> Max Bandwidth: " << (prop.memoryBusWidth/8*2*prop.memoryClockRate/1000000) << " GB/s" << endl;
        cout << "\tConstant Memory : " << prop.totalConstMem/1024 << " KB" << endl;
        cout << "\tShared Memory per Block: " << prop.sharedMemPerBlock/1024 << " KB" << endl;
        cout << "\tUnified Memory Addressing: " << (prop.unifiedAddressing ? "true" : "false") << endl;

        cout << "=== THREAD ===" << endl;
        cout << "\tMax Threads/SM: " << prop.maxThreadsPerMultiProcessor << endl;
        cout << "\tThreads/Block: " << prop.maxThreadsPerBlock << endl;
        cout << "\tMax Thread Size: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
        cout << "\tWarp Size: " << prop.warpSize << endl;
        cout << "\tMax Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
        cout << "\tRegisters/SM: " << prop.regsPerMultiprocessor/1024 << " KB" << endl;
        cout << "\tRegisters/Block: " << prop.regsPerBlock/1024 << " KB" << endl;


        cout << "=== TEXTURE ===" << endl;
        cout << "\tTexture Size 1D: " << prop.maxTexture1D  << endl;
        cout << "\tTexture Size 2D: " << prop.maxTexture2D[0] << " x " << prop.maxTexture2D[1] << endl;
        cout << "\tTexture Size 3D: " << prop.maxTexture3D[0] << " x " << prop.maxTexture3D[1] << " x " << prop.maxTexture3D[2] << endl;

    }
    return 0;
}
