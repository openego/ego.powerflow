from tools.pypsa_io import oedb_session, get_pq_sets,\
    get_timerange, import_components, import_pq_sets, create_powerflow_problem,\
    add_coordinates, plot_line_loading

from egoio.db_tables.calc_ego_hv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution, GeneratorPqSet, LoadPqSet

session = oedb_session()

scenario = 'Status Quo'

# define relevant tables of generator table
pq_set_cols = ['p_set']


# choose temp_id
temp_id_set = 1
start_h = 1
end_h = 50


# define investigated time range
timerange, slicer = get_timerange(session, temp_id_set, TempResolution, start_h, end_h)


# define relevant tables
tables = [Bus, Line, Generator, Load, Transformer]

# get components from database tables
components = import_components(tables, session, scenario)

# create PyPSA powerflow problem
network, snapshots = create_powerflow_problem(timerange, components)

# import pq-set tables to pypsa network
pq_object = [GeneratorPqSet, LoadPqSet]
network = import_pq_sets(session,
                         network,
                         pq_object,
                         timerange,
                         scenario, 
                         columns=pq_set_cols,                         
                         slicer = slicer)

# add coordinates to network nodes and make ready for map plotting
network = add_coordinates(network)

# start powerflow calculations
network.lpf(snapshots)

# make a line loading plot
plot_line_loading(network, output='show')


# close session
session.close()