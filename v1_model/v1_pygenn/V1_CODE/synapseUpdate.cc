#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedPresynapticUpdateGroup0
 {
    float* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    scalar g;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[4];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, float* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons, scalar g) {
    MergedPresynapticUpdateGroup0 group = {inSyn, srcSpkCnt, srcSpk, rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, g, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, 64, 96, 160, };
extern "C" __global__ void updatePresynapticKernel(float t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shRowLength[32];
    __shared__ unsigned int shSpk[32];
    // merged0
    if(id < 192) {
        unsigned int lo = 0;
        unsigned int hi = 4;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedPresynapticUpdateGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedPresynapticUpdateGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            atomicAdd(&group->inSyn[ipost], group->g);
                        }
                    }
                }
            }
        }
        
    }
}
void updateSynapses(float t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(6, 1);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
