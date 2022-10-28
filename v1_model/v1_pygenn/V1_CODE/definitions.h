#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <random>
#include <string>
#include <stdexcept>

// Standard C includes
#include <cassert>
#include <cstdint>
#define DT 1.00000000000000000e+00f
typedef float scalar;
#define SCALAR_MIN 1.175494351e-38f
#define SCALAR_MAX 3.402823466e+38f

#define TIME_MIN 1.175494351e-38f
#define TIME_MAX 3.402823466e+38f

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR float t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_E glbSpkCntE[0]
#define spike_E glbSpkE
#define glbSpkShiftE 0

EXPORT_VAR unsigned int* glbSpkCntE;
EXPORT_VAR unsigned int* d_glbSpkCntE;
EXPORT_VAR unsigned int* glbSpkE;
EXPORT_VAR unsigned int* d_glbSpkE;
EXPORT_VAR uint32_t* recordSpkE;
EXPORT_VAR uint32_t* d_recordSpkE;
EXPORT_VAR scalar* VE;
EXPORT_VAR scalar* d_VE;
EXPORT_VAR unsigned int* SpikeCountE;
EXPORT_VAR unsigned int* d_SpikeCountE;
#define spikeCount_I glbSpkCntI[0]
#define spike_I glbSpkI
#define glbSpkShiftI 0

EXPORT_VAR unsigned int* glbSpkCntI;
EXPORT_VAR unsigned int* d_glbSpkCntI;
EXPORT_VAR unsigned int* glbSpkI;
EXPORT_VAR unsigned int* d_glbSpkI;
EXPORT_VAR uint32_t* recordSpkI;
EXPORT_VAR uint32_t* d_recordSpkI;
EXPORT_VAR scalar* VI;
EXPORT_VAR scalar* d_VI;
EXPORT_VAR unsigned int* SpikeCountI;
EXPORT_VAR unsigned int* d_SpikeCountI;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR float* inSynIE;
EXPORT_VAR float* d_inSynIE;
EXPORT_VAR float* inSynEE;
EXPORT_VAR float* d_inSynEE;
EXPORT_VAR float* inSynII;
EXPORT_VAR float* d_inSynII;
EXPORT_VAR float* inSynEI;
EXPORT_VAR float* d_inSynEI;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthEE;
EXPORT_VAR unsigned int* rowLengthEE;
EXPORT_VAR unsigned int* d_rowLengthEE;
EXPORT_VAR uint32_t* indEE;
EXPORT_VAR uint32_t* d_indEE;
EXPORT_VAR const unsigned int maxRowLengthEI;
EXPORT_VAR unsigned int* rowLengthEI;
EXPORT_VAR unsigned int* d_rowLengthEI;
EXPORT_VAR uint32_t* indEI;
EXPORT_VAR uint32_t* d_indEI;
EXPORT_VAR const unsigned int maxRowLengthIE;
EXPORT_VAR unsigned int* rowLengthIE;
EXPORT_VAR unsigned int* d_rowLengthIE;
EXPORT_VAR uint32_t* indIE;
EXPORT_VAR uint32_t* d_indIE;
EXPORT_VAR const unsigned int maxRowLengthII;
EXPORT_VAR unsigned int* rowLengthII;
EXPORT_VAR unsigned int* d_rowLengthII;
EXPORT_VAR uint32_t* indII;
EXPORT_VAR uint32_t* d_indII;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

EXPORT_FUNC void pushESpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullESpikesFromDevice();
EXPORT_FUNC void pushECurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullECurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getECurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getECurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVEFromDevice();
EXPORT_FUNC void pushCurrentVEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVEFromDevice();
EXPORT_FUNC scalar* getCurrentVE(unsigned int batch = 0); 
EXPORT_FUNC void pushSpikeCountEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeCountEFromDevice();
EXPORT_FUNC void pushCurrentSpikeCountEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentSpikeCountEFromDevice();
EXPORT_FUNC unsigned int* getCurrentSpikeCountE(unsigned int batch = 0); 
EXPORT_FUNC void pushEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEStateFromDevice();
EXPORT_FUNC void pushISpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullISpikesFromDevice();
EXPORT_FUNC void pushICurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullICurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getICurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getICurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVIFromDevice();
EXPORT_FUNC void pushCurrentVIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVIFromDevice();
EXPORT_FUNC scalar* getCurrentVI(unsigned int batch = 0); 
EXPORT_FUNC void pushSpikeCountIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullSpikeCountIFromDevice();
EXPORT_FUNC void pushCurrentSpikeCountIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentSpikeCountIFromDevice();
EXPORT_FUNC unsigned int* getCurrentSpikeCountI(unsigned int batch = 0); 
EXPORT_FUNC void pushIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIStateFromDevice();
EXPORT_FUNC void pushEEConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEEConnectivityFromDevice();
EXPORT_FUNC void pushEIConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEIConnectivityFromDevice();
EXPORT_FUNC void pushIEConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIEConnectivityFromDevice();
EXPORT_FUNC void pushIIConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIIConnectivityFromDevice();
EXPORT_FUNC void pushinSynEEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEEFromDevice();
EXPORT_FUNC void pushEEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEEStateFromDevice();
EXPORT_FUNC void pushinSynEIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynEIFromDevice();
EXPORT_FUNC void pushEIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullEIStateFromDevice();
EXPORT_FUNC void pushinSynIEToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynIEFromDevice();
EXPORT_FUNC void pushIEStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIEStateFromDevice();
EXPORT_FUNC void pushinSynIIToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynIIFromDevice();
EXPORT_FUNC void pushIIStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullIIStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateRecordingBuffers(unsigned int timesteps);
EXPORT_FUNC void pullRecordingBuffersFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(float t, unsigned int recordingTimestep); 
EXPORT_FUNC void updateSynapses(float t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
