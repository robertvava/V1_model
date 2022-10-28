from pygenn.genn_model import GeNNModel, init_var, init_connectivity
import numpy as np
import matplotlib.pyplot as plt
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class)
from os import path
from pygenn.genn_wrapper import NO_DELAY


"""
VARIABLE DESCRITPIONS:

V(t) = Membrane Potential
I_j(t) =   After-spike currents
Theta_s(t) = Spike-dependent threshold component
Theta_v(t) = Voltage-dependent threshold component

PARAMTER DESCRIPTIONS:                                                  FIT FROM:  

C = Capacitance                                                 <-      Sub-threshold nosie
R = Membrane Resistance                                         <-      Sub-threshold nosie
E_L = Resting Potential                                         <-      Resting V before noise
Theta_inf = Instantaneous Threshold                             <-      Short Square input
delta t = Spike cut length                                      <-      All noise spikes
f_v = Voltage Fraction following spike                          <-      All noise spike
delta V = Voltage addition following spike                      <-      All noise spike
b_s = Spike-induced threshold time constant                     <-      Triple short square
delta Theta_s = Threshold addition following spike              <-      Triple short square
delta I_j = After-spike current amplitudes                      <-      Supra-threshold noise
k_j = After-spike current time constants                        <-      Supra-threshold noise
f_v = Current fraction following spike                          <-      Set to 1
a_v = Adaptation index of threshold                             <-      Prespike V supra-thr. noise
b_v = Voltage-induced threshold time constant                   <-      Prespike V supra-thr. noise

"""

glif1 = create_custom_neuron_class(
    "GLIF1", # Or simply LIF

    param_names=["C", "G", "E_L", "I_e", "Vthr", "Vreset"],    # Vthr = Theta inf
    var_name_types=[("V", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code = "$(V) += (1/$(C)*($(I_e) - $(G)*( $(V) - $(E_L)))) * DT;",
    threshold_condition_code="$(V) > $(Vthr)",
    reset_code= """
    $(V) = $(Vreset);    
    $(SpikeCount)++;
    """
    )

glif2 = create_custom_neuron_class(
    "GLIF2", # Also called LIF-R (biologically defined reset rules)
    
    param_names=["C", "G", "E_L", "I_e", "Vthr", "Vrest", "f_v", "dO", "dV", "b"],
    var_name_types=[("V", "scalar"), ("Theta", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code = """
    $(V) += (1/$(C) * ($(I_e) - $(G) * ($(V) - $(E_L)))) * DT;
    $(theta) += (-$(b) * $(theta)) * DT;    
    """,
    threshold_condition_code="$(V) >= $(Vthr) + $(theta)",
    reset_code= """
    $(V) = $(E_L) + $(f_v) * ($(V) - $(E_L)) - $(dV);
    $(theta) = $(theta) + $(dTheta);
    $(SpikeCount)++;    
    """
)

glif3 = create_custom_neuron_class(
    "GLIF3", # Also called LIF-ASC (LIF with after-spike current)


    param_names=["C", "G", "E_L", "I_e", "Vthr", "Vrest", "k_j", "R_j", "A_j"],
    var_name_types=[("V", "scalar"), ("I_j", "scalar"), ("SpikeCount", "unsigned int"), ("f_v", "unsigned int"), ("dV", "scalar"), ("O", 'unsigned int'), ("dO", "unsigned int")],
    sim_code = """     
    $(I_j) = -$(I_j)
    $(V) += (1/$(C)*($(I_e) - $(G)*( $(V) - $(E_L)))) * DT;
    """,
    threshold_condition_code="$(V) >= $(Vthr) + $(k_j)",
    reset_code= """
    $(V) = $(E_L) + $(f_v) * ($(V) - $(E_L)) - $(dV);
    $(O) = $(O) + $(dO);
    $(SpikeCount)++;    
    """ 
)

glif4 = create_custom_neuron_class(
    "GLIF4",

    param_names=["C", "G", "E_L", "I_e", "Vthr", "Vrest"],
    var_name_types=[("V", "scalar"), ("I_j", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code = """
    $(V) += (1/$(C)*($(I_e) - $(G)*( $(V) - $(E_L)))) * DT;
    """,
    threshold_condition_code="$(V) >= $(Vthr) + $(O)",
    reset_code= """
    $(V) = $(E_L) + $(f_v) * ($(V) - $(E_L)) - $(dV);
    $(O) = $(O) + $(dO);
    $(SpikeCount)++;
    """
)

glif5 = create_custom_neuron_class(
    "GLIF5",

    param_names=["C", "G", "E_L", "I_e", "Vthr", "Vrest"],
    var_name_types=[("V", "scalar"), ("I_j", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code = """
    $(V) += (1/$(C)*($(I_e) - $(G)*( $(V) - $(E_L)))) * DT;
    """,
    threshold_condition_code="$(V) >= $(Vthr) + $(O)",
    reset_code= """
    $(V) = $(E_L) + $(f_v) * ($(V) - $(E_L)) - $(dV);
    $(O) = $(O) + $(dO);
    $(SpikeCount)++;
    """
)




# ====================================== GLIF 1 ====================================== # 

glif1_params = {"C": 1, "G": 4.6951, "Vreset": -50,
              "Vthr": 26.5, "E_L": -77.4, "I_e": 1.0}   

glif1_init = {"V": init_var("Uniform", {"min": -77.4, "max": -50.0}),
            "SpikeCount": 0}

# ====================================== GLIF 1 ====================================== # 



# ------------------------------------------------------------------------------------ #



# ====================================== GLIF 2 ====================================== # 

glif2_params = {"C": 1.0, "G": 10.0, "Vrest": -49.0, "dO": 1.2, "b": 1,
              "Vthr": -55.0, "E_L": 2.5, "I_e": 5.0, "f_v": 1.1, "dV" : 1.3}   

glif2_init = {"V": init_var("Uniform", {"min": -60.0, "max": -50.0}),
            "O": init_var("Uniform", {"min": 10, "max": 20}),
            "SpikeCount": 0}   

# ====================================== GLIF 2 ====================================== # 



# ------------------------------------------------------------------------------------ #



# ====================================== GLIF 3 ====================================== # 

glif3_params = {"C": 1.0, "G": 10.0, "Vrest": -49.0, "R_j": 1.0,
                "k_j" : 1.0 }   

glif3_init = {"V": init_var("Uniform", {"min": -60.0, "max": -50.0}),
            "O": init_var("Uniform", {"min": 10, "max": 20}),
            "SpikeCount": 0}   

glif3_params = {'A_j': 1, 'C': 1, 'E_L': 0.55, 'G': 0.24, 'I_e': 1, 'R_j': 0.5, 'Vrest': -55, 'Vthr': -70, 'k_j': 1}

glif3_init = {'I_j': 1, 'SpikeCount': 1, 'V': 1, "O": 1, "dO": 0.001, "dV":0.005, "f_v":1}

# ====================================== GLIF 3 ====================================== # 



# ------------------------------------------------------------------------------------ #



# ====================================== GLIF 4 ====================================== # 

glif4_params = {"C": 1.0, "G": 10.0, "Vrest": -49.0, "dO": 1.2, "b": 1,
              "Vthr": -55.0, "E_L": 2.5, "I_e": 5.0, "f_v": 1.1, "dV" : 1.3}   

glif4_init = {"V": init_var("Uniform", {"min": -60.0, "max": -50.0}),
            "O": init_var("Uniform", {"min": 10, "max": 20}),
            "SpikeCount": 0}   

# ====================================== GLIF 4 ====================================== # 



# ------------------------------------------------------------------------------------ #



# ====================================== GLIF 5 ====================================== # 

glif5_params = {"C": 1.0, "G": 10.0, "Vrest": -49.0, "dO": 1.2, "b": 1,
              "Vthr": -55.0, "E_L": 2.5, "I_e": 5.0, "f_v": 1.1, "dV" : 1.3}   

glif5_init = {"V": init_var("Uniform", {"min": -60.0, "max": -50.0}),
            "O": init_var("Uniform", {"min": 10, "max": 20}),
            "SpikeCount": 0}   

# ====================================== GLIF 5 ====================================== # 


# ------------------------------------------------------------------------------------ # 


# =========================== Building blocks for V1 model =========================== #


v1_model = GeNNModel("float", "V1")
v1_model.dT = 1.0


exc_pop = v1_model.add_neuron_population("E", 3200, glif1, glif1_params, glif1_init)
inh_pop = v1_model.add_neuron_population("I", 800, glif1, glif1_params, glif1_init)


exc_pop.spike_recording_enabled = True
inh_pop.spike_recording_enabled = True

exc_synapse_init = {"g": 0.0008}
inh_synapse_init = {"g": -0.0102}

exc_post_syn_params = {"tau": 5.0}
inh_post_syn_params = {"tau": 10.0}

fixed_prob = {"prob": 0.1}

v1_model.add_synapse_population("EE", "SPARSE_GLOBALG", 0,
    exc_pop, exc_pop,
    "StaticPulse", {}, exc_synapse_init, {}, {},
    "ExpCurr", exc_post_syn_params, {},
    init_connectivity("FixedProbabilityNoAutapse", fixed_prob))

v1_model.add_synapse_population("EI", "SPARSE_GLOBALG", 0,
    exc_pop, inh_pop,
    "StaticPulse", {}, exc_synapse_init, {}, {},
    "ExpCurr", exc_post_syn_params, {},
    init_connectivity("FixedProbability", fixed_prob))

v1_model.add_synapse_population("II", "SPARSE_GLOBALG", 0,
    inh_pop, inh_pop,
    "StaticPulse", {}, inh_synapse_init, {}, {},
    "ExpCurr", inh_post_syn_params, {},
    init_connectivity("FixedProbabilityNoAutapse", fixed_prob))

v1_model.add_synapse_population("IE", "SPARSE_GLOBALG", 0,
    inh_pop, exc_pop,
    "StaticPulse", {}, inh_synapse_init, {}, {},
    "ExpCurr", inh_post_syn_params, {},
    init_connectivity("FixedProbability", fixed_prob))



v1_model.build()
v1_model.load(num_recording_timesteps=1000)

while v1_model.timestep < 1000:
    v1_model.step_time()

v1_model.pull_recording_buffers_from_device()

exc_spike_times, exc_spike_ids = exc_pop.spike_recording_data
inh_spike_times, inh_spike_ids = inh_pop.spike_recording_data


fig, axes = plt.subplots(3, sharex=True, figsize=(20, 10))

# Define some bins to calculate spike rates
bin_size = 20.0
rate_bins = np.arange(0, 1000.0, bin_size)
rate_bin_centres = rate_bins[:-1] + (bin_size / 2.0)

# Plot excitatory and inhibitory spikes on first axis
axes[0].scatter(exc_spike_times, exc_spike_ids, s=1)
axes[0].scatter(inh_spike_times, inh_spike_ids + 320, s=1)

# Plot excitatory rates on second axis
exc_rate = np.histogram(exc_spike_times, bins=rate_bins)[0]
axes[1].plot(rate_bin_centres, exc_rate * (1000.0 / bin_size) * (1.0 / 3200.0))

# Plot inhibitory rates on third axis
inh_rate = np.histogram(inh_spike_times, bins=rate_bins)[0]
axes[2].plot(rate_bin_centres, inh_rate * (1000.0 / bin_size) * (1.0 / 800.0))

# Label axes
axes[0].set_ylabel("Neuron ID")
axes[1].set_ylabel("Excitatory rate[Hz]")
axes[2].set_ylabel("Inhibitory rate[Hz]")
axes[2].set_xlabel("Time [ms]")

plt.show()