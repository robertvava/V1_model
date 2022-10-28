#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
float t;
unsigned long long numRecordingTimesteps = 0;
__device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntE;
unsigned int* d_glbSpkCntE;
unsigned int* glbSpkE;
unsigned int* d_glbSpkE;
uint32_t* recordSpkE;
uint32_t* d_recordSpkE;
scalar* VE;
scalar* d_VE;
unsigned int* SpikeCountE;
unsigned int* d_SpikeCountE;
unsigned int* glbSpkCntI;
unsigned int* d_glbSpkCntI;
unsigned int* glbSpkI;
unsigned int* d_glbSpkI;
uint32_t* recordSpkI;
uint32_t* d_recordSpkI;
scalar* VI;
scalar* d_VI;
unsigned int* SpikeCountI;
unsigned int* d_SpikeCountI;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
float* inSynIE;
float* d_inSynIE;
float* inSynEE;
float* d_inSynEE;
float* inSynII;
float* d_inSynII;
float* inSynEI;
float* d_inSynEI;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthEE = 62;
unsigned int* rowLengthEE;
unsigned int* d_rowLengthEE;
uint32_t* indEE;
uint32_t* d_indEE;
const unsigned int maxRowLengthEI = 24;
unsigned int* rowLengthEI;
unsigned int* d_rowLengthEI;
uint32_t* indEI;
uint32_t* d_indEI;
const unsigned int maxRowLengthIE = 60;
unsigned int* rowLengthIE;
unsigned int* d_rowLengthIE;
uint32_t* indIE;
uint32_t* d_indIE;
const unsigned int maxRowLengthII = 23;
unsigned int* rowLengthII;
unsigned int* d_rowLengthII;
uint32_t* indII;
uint32_t* d_indII;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushESpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntE, glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkE, glbSpkE, 320 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushECurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntE, glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkE, glbSpkE, glbSpkCntE[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VE, VE, 320 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVEToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VE, VE, 320 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushSpikeCountEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_SpikeCountE, SpikeCountE, 320 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushCurrentSpikeCountEToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_SpikeCountE, SpikeCountE, 320 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushEStateToDevice(bool uninitialisedOnly) {
    pushVEToDevice(uninitialisedOnly);
    pushSpikeCountEToDevice(uninitialisedOnly);
}

void pushISpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntI, glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkI, glbSpkI, 80 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushICurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntI, glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkI, glbSpkI, glbSpkCntI[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VI, VI, 80 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushCurrentVIToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VI, VI, 80 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushSpikeCountIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_SpikeCountI, SpikeCountI, 80 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushCurrentSpikeCountIToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_SpikeCountI, SpikeCountI, 80 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushIStateToDevice(bool uninitialisedOnly) {
    pushVIToDevice(uninitialisedOnly);
    pushSpikeCountIToDevice(uninitialisedOnly);
}

void pushEEConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthEE, rowLengthEE, 320 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indEE, indEE, 19840 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushEIConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthEI, rowLengthEI, 320 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indEI, indEI, 7680 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushIEConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthIE, rowLengthIE, 80 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indIE, indIE, 4800 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushIIConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthII, rowLengthII, 80 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indII, indII, 1840 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushinSynEEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynEE, inSynEE, 320 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushEEStateToDevice(bool uninitialisedOnly) {
    pushinSynEEToDevice(uninitialisedOnly);
}

void pushinSynEIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynEI, inSynEI, 80 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushEIStateToDevice(bool uninitialisedOnly) {
    pushinSynEIToDevice(uninitialisedOnly);
}

void pushinSynIEToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynIE, inSynIE, 320 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushIEStateToDevice(bool uninitialisedOnly) {
    pushinSynIEToDevice(uninitialisedOnly);
}

void pushinSynIIToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynII, inSynII, 80 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushIIStateToDevice(bool uninitialisedOnly) {
    pushinSynIIToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pullESpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntE, d_glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkE, d_glbSpkE, 320 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullECurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntE, d_glbSpkCntE, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkE, d_glbSpkE, glbSpkCntE[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VE, d_VE, 320 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VE, d_VE, 320 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullSpikeCountEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(SpikeCountE, d_SpikeCountE, 320 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullCurrentSpikeCountEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(SpikeCountE, d_SpikeCountE, 320 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullEStateFromDevice() {
    pullVEFromDevice();
    pullSpikeCountEFromDevice();
}

void pullISpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntI, d_glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkI, d_glbSpkI, 80 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullICurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntI, d_glbSpkCntI, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkI, d_glbSpkI, glbSpkCntI[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VI, d_VI, 80 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(VI, d_VI, 80 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullSpikeCountIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(SpikeCountI, d_SpikeCountI, 80 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullCurrentSpikeCountIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(SpikeCountI, d_SpikeCountI, 80 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullIStateFromDevice() {
    pullVIFromDevice();
    pullSpikeCountIFromDevice();
}

void pullEEConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthEE, d_rowLengthEE, 320 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indEE, d_indEE, 19840 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullEIConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthEI, d_rowLengthEI, 320 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indEI, d_indEI, 7680 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullIEConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthIE, d_rowLengthIE, 80 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indIE, d_indIE, 4800 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullIIConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthII, d_rowLengthII, 80 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indII, d_indII, 1840 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullinSynEEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynEE, d_inSynEE, 320 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullEEStateFromDevice() {
    pullinSynEEFromDevice();
}

void pullinSynEIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynEI, d_inSynEI, 80 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullEIStateFromDevice() {
    pullinSynEIFromDevice();
}

void pullinSynIEFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynIE, d_inSynIE, 320 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullIEStateFromDevice() {
    pullinSynIEFromDevice();
}

void pullinSynIIFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynII, d_inSynII, 80 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullIIStateFromDevice() {
    pullinSynIIFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getECurrentSpikes(unsigned int batch) {
    return (glbSpkE);
}

unsigned int& getECurrentSpikeCount(unsigned int batch) {
    return glbSpkCntE[0];
}

scalar* getCurrentVE(unsigned int batch) {
    return VE;
}

unsigned int* getCurrentSpikeCountE(unsigned int batch) {
    return SpikeCountE;
}

unsigned int* getICurrentSpikes(unsigned int batch) {
    return (glbSpkI);
}

unsigned int& getICurrentSpikeCount(unsigned int batch) {
    return glbSpkCntI[0];
}

scalar* getCurrentVI(unsigned int batch) {
    return VI;
}

unsigned int* getCurrentSpikeCountI(unsigned int batch) {
    return SpikeCountI;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushEStateToDevice(uninitialisedOnly);
    pushIStateToDevice(uninitialisedOnly);
    pushEEStateToDevice(uninitialisedOnly);
    pushEIStateToDevice(uninitialisedOnly);
    pushIEStateToDevice(uninitialisedOnly);
    pushIIStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushEEConnectivityToDevice(uninitialisedOnly);
    pushEIConnectivityToDevice(uninitialisedOnly);
    pushIEConnectivityToDevice(uninitialisedOnly);
    pushIIConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pullEStateFromDevice();
    pullIStateFromDevice();
    pullEEStateFromDevice();
    pullEIStateFromDevice();
    pullIEStateFromDevice();
    pullIIStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pullECurrentSpikesFromDevice();
    pullICurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateRecordingBuffers(unsigned int timesteps) {
    numRecordingTimesteps = timesteps;
     {
        const unsigned int numWords = 10 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkE, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkE, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate0recordSpkToDevice(0, d_recordSpkE);
        }
    }
     {
        const unsigned int numWords = 3 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkI, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkI, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate0recordSpkToDevice(1, d_recordSpkI);
        }
    }
}

void pullRecordingBuffersFromDevice() {
    if(numRecordingTimesteps == 0) {
        throw std::runtime_error("Recording buffer not allocated - cannot pull from device");
    }
     {
        const unsigned int numWords = 10 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkE, d_recordSpkE, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 3 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkI, d_recordSpkI, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
}

void allocateMem() {
    int deviceID;
    CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, "0000:58:00.0"));
    CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
    
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntE, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntE, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkE, 320 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkE, 320 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&VE, 320 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_VE, 320 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&SpikeCountE, 320 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_SpikeCountE, 320 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntI, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntI, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkI, 80 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkI, 80 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&VI, 80 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_VI, 80 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&SpikeCountI, 80 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_SpikeCountI, 80 * sizeof(unsigned int)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynIE, 320 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynIE, 320 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynEE, 320 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynEE, 320 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynII, 80 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynII, 80 * sizeof(float)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynEI, 80 * sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynEI, 80 * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthEE, 320 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthEE, 320 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indEE, 19840 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indEE, 19840 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthEI, 320 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthEI, 320 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indEI, 7680 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indEI, 7680 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthIE, 80 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthIE, 80 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indIE, 4800 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indIE, 4800 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthII, 80 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthII, 80 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indII, 1840 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indII, 1840 * sizeof(uint32_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntE, d_glbSpkE, d_VE, d_SpikeCountE, d_inSynIE, d_inSynEE, 320);
    pushMergedNeuronInitGroup0ToDevice(1, d_glbSpkCntI, d_glbSpkI, d_VI, d_SpikeCountI, d_inSynII, d_inSynEI, 80);
    pushMergedSynapseConnectivityInitGroup0ToDevice(0, d_rowLengthEI, d_indEI, 24, 320, 80);
    pushMergedSynapseConnectivityInitGroup0ToDevice(1, d_rowLengthIE, d_indIE, 60, 80, 320);
    pushMergedSynapseConnectivityInitGroup1ToDevice(0, d_rowLengthEE, d_indEE, 62, 320, 320);
    pushMergedSynapseConnectivityInitGroup1ToDevice(1, d_rowLengthII, d_indII, 23, 80, 80);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntE, d_glbSpkE, d_VE, d_SpikeCountE, d_inSynIE, d_inSynEE, d_recordSpkE, 320);
    pushMergedNeuronUpdateGroup0ToDevice(1, d_glbSpkCntI, d_glbSpkI, d_VI, d_SpikeCountI, d_inSynII, d_inSynEI, d_recordSpkI, 80);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynEE, d_glbSpkCntE, d_glbSpkE, d_rowLengthEE, d_indEE, 62, 320, 320, 8.00000000000000038e-04f);
    pushMergedPresynapticUpdateGroup0ToDevice(1, d_inSynEI, d_glbSpkCntE, d_glbSpkE, d_rowLengthEI, d_indEI, 24, 320, 80, 8.00000000000000038e-04f);
    pushMergedPresynapticUpdateGroup0ToDevice(2, d_inSynIE, d_glbSpkCntI, d_glbSpkI, d_rowLengthIE, d_indIE, 60, 80, 320, -1.02000000000000007e-02f);
    pushMergedPresynapticUpdateGroup0ToDevice(3, d_inSynII, d_glbSpkCntI, d_glbSpkI, d_rowLengthII, d_indII, 23, 80, 80, -1.02000000000000007e-02f);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntE);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(1, d_glbSpkCntI);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntE));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntE));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkE));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkE));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkE));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkE));
    CHECK_CUDA_ERRORS(cudaFreeHost(VE));
    CHECK_CUDA_ERRORS(cudaFree(d_VE));
    CHECK_CUDA_ERRORS(cudaFreeHost(SpikeCountE));
    CHECK_CUDA_ERRORS(cudaFree(d_SpikeCountE));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntI));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkI));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkI));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkI));
    CHECK_CUDA_ERRORS(cudaFreeHost(VI));
    CHECK_CUDA_ERRORS(cudaFree(d_VI));
    CHECK_CUDA_ERRORS(cudaFreeHost(SpikeCountI));
    CHECK_CUDA_ERRORS(cudaFree(d_SpikeCountI));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynIE));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynIE));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynEE));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynII));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynII));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynEI));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynEI));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthEE));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(indEE));
    CHECK_CUDA_ERRORS(cudaFree(d_indEE));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthEI));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(indEI));
    CHECK_CUDA_ERRORS(cudaFree(d_indEI));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthIE));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthIE));
    CHECK_CUDA_ERRORS(cudaFreeHost(indIE));
    CHECK_CUDA_ERRORS(cudaFree(d_indIE));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthII));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthII));
    CHECK_CUDA_ERRORS(cudaFreeHost(indII));
    CHECK_CUDA_ERRORS(cudaFree(d_indII));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    
}

size_t getFreeDeviceMemBytes() {
    size_t free;
    size_t total;
    CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));
    return free;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t, (unsigned int)(iT % numRecordingTimesteps)); 
    iT++;
    t = iT*DT;
}

