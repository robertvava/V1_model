from bmtk.analyzer.spike_trains import plot_raster, plot_rates
from bmtk.analyzer.compartment import plot_traces

_= plot_raster(config_file='sim_ch05/config.json', group_by='pop_name', plt_style='seaborn-muted')

_= plot_rates(config_file='sim_ch05/config.json', group_by='pop_name', plt_style='seaborn-muted')   

_ = plot_traces(config_file='sim_ch05/config.json', group_by='pop_name', plt_style='seaborn-muted')