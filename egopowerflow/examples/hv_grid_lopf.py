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
from egopowerflow.tools.io import get_timerange, import_components, import_pq_sets,\
    add_source_types, create_powerflow_problem
from egopowerflow.tools.plot import add_coordinates, plot_line_loading,\
     plot_stacked_gen
from egoio.db_tables.calc_ego_hv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution, GeneratorPqSet, LoadPqSet, Source

session = oedb_session()

scenario = 'Status Quo'

# define relevant tables of generator table
pq_set_cols_1 = ['p_set']
pq_set_cols_2 = ['q_set']
p_max_pu = ['p_max_pu']
# choose relevant parameters used in pf
temp_id_set = 1
start_h = 500
end_h = 524

# define investigated time range
timerange = get_timerange(session, temp_id_set, TempResolution, start_h, end_h)

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
                         columns=pq_set_cols_1,                         
                         start_h=start_h,
                         end_h=end_h)

# import pq-set table to pypsa network (q_set for loads)
network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=[LoadPqSet],
                         timerange=timerange,
                         scenario=scenario, 
                         columns=pq_set_cols_2,                         
                         start_h=start_h,
                         end_h=end_h)
                         
network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=[GeneratorPqSet],
                         timerange=timerange,
                         scenario=scenario, 
                         columns=p_max_pu,                         
                         start_h=start_h,
                         end_h=end_h)                         

# add coordinates to network nodes and make ready for map plotting
network = add_coordinates(network)

# add source names to generators
add_source_types(session, network, table=Source)

# start powerflow calculations
network.lopf(snapshots)
network.model.write('file.lp', io_options={'symbolic_labels':True})

# make a line loading plot
plot_line_loading(network, output='show')

#plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# same as before, limited to one specific bus
plot_stacked_gen(network, bus='24560', resolution='MW')

# close session
session.close()