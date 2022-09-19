import numpy as np
import matplotlib.pyplot as plt
from pygenn.genn_model import GeNNModel, init_connectivity, init_var





model = GeNNModel("float", "tutorial2")
model.dT = 1.0

lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -49.0, "Vreset": -60.0,
              "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 5.0}   

lif_init = {"V": init_var("Uniform", {"min": -60.0, "max": -50.0}),
            "RefracTime": 0.0}

exc_pop = model.add_neuron_population("E", 3200, "LIF", lif_params, lif_init)
inh_pop = model.add_neuron_population("I", 800, "LIF", lif_params, lif_init)

exc_pop.spike_recording_enabled = True
inh_pop.spike_recording_enabled = True

exc_synapse_init = {"g": 0.0008}
inh_synapse_init = {"g": -0.0102}

exc_post_syn_params = {"tau": 5.0}
inh_post_syn_params = {"tau": 10.0}

fixed_prob = {"prob": 0.1}

model.add_synapse_population("EE", "SPARSE_GLOBALG", 0,
    exc_pop, exc_pop,
    "StaticPulse", {}, exc_synapse_init, {}, {},
    "ExpCurr", exc_post_syn_params, {},
    init_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("EI", "SPARSE_GLOBALG", 0,
    exc_pop, inh_pop,
    "StaticPulse", {}, exc_synapse_init, {}, {},
    "ExpCurr", exc_post_syn_params, {},
    init_connectivity("FixedProbability", fixed_prob))

model.add_synapse_population("II", "SPARSE_GLOBALG", 0,
    inh_pop, inh_pop,
    "StaticPulse", {}, inh_synapse_init, {}, {},
    "ExpCurr", inh_post_syn_params, {},
    init_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("IE", "SPARSE_GLOBALG", 0,
    inh_pop, exc_pop,
    "StaticPulse", {}, inh_synapse_init, {}, {},
    "ExpCurr", inh_post_syn_params, {},
    init_connectivity("FixedProbability", fixed_prob))

model.build()
model.load(num_recording_timesteps=1000)


while model.timestep < 1000:
    model.step_time()

model.pull_recording_buffers_from_device()

exc_spike_times, exc_spike_ids = exc_pop.spike_recording_data
inh_spike_times, inh_spike_ids = inh_pop.spike_recording_data

fig, axes = plt.subplots(3, sharex=True, figsize=(20, 10))

# Define some bins to calculate spike rates
bin_size = 20.0
rate_bins = np.arange(0, 1000.0, bin_size)
rate_bin_centres = rate_bins[:-1] + (bin_size / 2.0)

# Plot excitatory and inhibitory spikes on first axis
axes[0].scatter(exc_spike_times, exc_spike_ids, s=1)
axes[0].scatter(inh_spike_times, inh_spike_ids + 3200, s=1)

# Plot excitatory rates on second axis
exc_rate = np.histogram(exc_spike_times, bins=rate_bins)[0]
axes[1].plot(rate_bin_centres, exc_rate * (1000.0 / bin_size) * (1.0 / 3200.0))

# Plot inhibitory rates on third axis
inh_rate = np.histogram(inh_spike_times, bins=rate_bins)[0]
axes[2].plot(rate_bin_centres, inh_rate * (1000.0 / bin_size) * (1.0 / 800.0))

# Label axes
axes[0].set_ylabel("Neuron ID")
axes[1].set_ylabel("Excitatory rate [Hz]")
axes[2].set_ylabel("Inhibitory rate [Hz]")
axes[2].set_xlabel("Time [ms]")

plt.show()