from egopowerflow.tools.pypsa_io import oedb_session, get_pq_sets,\
    get_timerange, import_components, import_pq_sets, create_powerflow_problem,\
    add_coordinates, plot_line_loading

from egoio.db_tables.calc_ego_mv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution, BusVMagSet, GeneratorPqSet, LoadPqSet


session = oedb_session()

# define relevant tables of generator table
pq_set_cols = ['temp_id', 'p_set', 'q_set']


# choose temp_id
temp_id_set = 1

# examplary call of pq-set retrieval
gen_pq_set = get_pq_sets(session, GeneratorPqSet, index_col='generator_id',
                         columns=pq_set_cols)
load_pq_set = get_pq_sets(session, LoadPqSet, index_col='load_id',
                          columns=pq_set_cols)
bus_vmag_set = get_pq_sets(session, BusVMagSet, index_col='bus_id',
                           columns=['temp_id', 'v_mag_pu_set'])

# define investigated time range
timerange = get_timerange(session, temp_id_set)

# define relevant tables
tables = [Bus, Line, Generator, Load, Transformer]

# get components from database tables
components = import_components(tables, session)

# create PyPSA powerflow problem
network, snapshots = create_powerflow_problem(timerange, components)

# import pq-set tables to pypsa network
pq_object = [GeneratorPqSet, LoadPqSet, BusVMagSet]
network = import_pq_sets(session,
                         network,
                         pq_object,
                         timerange)

# add coordinates to network nodes and make ready for map plotting
network = add_coordinates(network)

# start powerflow calculations
network.pf(snapshots)

# make a line loading plot
plot_line_loading(network, output='file')


# close session
session.close()