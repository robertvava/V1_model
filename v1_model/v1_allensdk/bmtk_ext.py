from bmtk.simulator import pointnet

# THIS IS THE FILE YOU NEED TO RUN AFTER SORTING OUT NEST

configure = pointnet.Config.from_json('sim_ch05/config.json')
configure.build_env()
network = pointnet.PointNetwork.from_config(configure)
sim = pointnet.PointSimulator.from_config(configure, network)
sim.run()