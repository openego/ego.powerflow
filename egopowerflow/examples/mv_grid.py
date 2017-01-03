"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.
Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line."""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "tba"


from egopowerflow.tools.tools import oedb_session
session = oedb_session(section='oedb')
# TODO: move oedb_session call below import statements
# this is currently not possible because, for some stupid reason and unknown, session creation then fails

from egopowerflow.tools.io import get_timerange, import_components, import_pq_sets,\
    add_source_types, create_powerflow_problem, get_pq_sets, results_to_oedb
from egopowerflow.tools.plot import add_coordinates, plot_line_loading


from egoio.db_tables.calc_ego_mv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution, BusVMagSet, GeneratorPqSet, LoadPqSet
from egoio.db_tables.calc_ego_mv_powerflow import TempResolution



scenario = 'Status Quo'

# define relevant tables of generator table
pq_set_cols = ['temp_id', 'p_set', 'q_set']

# choose temp_id
temp_id_set = 1

# define investigated time range
timerange = get_timerange(session, temp_id_set, TempResolution)

# define relevant tables
tables = [Bus, Line, Generator, Load, Transformer]

# get components from database tables
components = import_components(tables, session, scenario)

# create PyPSA powerflow problem
network, snapshots = create_powerflow_problem(timerange, components)

# import pq-set tables to pypsa network (p_set for generators and loads)
pq_object = [GeneratorPqSet, LoadPqSet]
network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=pq_object,
                         timerange=timerange,
                         scenario=scenario,
                         columns=['p_set'],
                         start_h=0,
                         end_h=2)

# import pq-set table to pypsa network (q_set for loads)
network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=pq_object,
                         timerange=timerange,
                         scenario=scenario,
                         columns=['q_set'],
                         start_h=0,
                         end_h=2)

# Import `v_mag_pu_set` for Bus
network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=[BusVMagSet],
                         timerange=timerange,
                         scenario=scenario,
                         columns=['v_mag_pu_set'],
                         start_h=0,
                         end_h=2)

# add coordinates to network nodes and make ready for map plotting
network = add_coordinates(network)

# start powerflow calculations
network.pf(snapshots)

# make a line loading plot
plot_line_loading(network, timestep=0, filename='Line_loading_load_case.png')
plot_line_loading(network, timestep=1, filename='Line_loading_feed-in_case.png')

results_to_oedb(session, network)

# close session
session.close()