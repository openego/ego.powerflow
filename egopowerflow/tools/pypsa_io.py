import pypsa
import pandas as pd

from sqlalchemy.orm import sessionmaker, load_only
from sqlalchemy import create_engine
from datetime import datetime
from pypsa import io
from math import sqrt
from geoalchemy2.shape import to_shape
from matplotlib import pyplot as plt


def oedb_session(section='oedb'):
    """Get SQLAlchemy session object with valid connection to OEDB"""

    # get session object by oemof.db tools (requires .oemof/config.ini
    try:
        from oemof import db
        conn = db.connection(section=section)

    except:
        print('Please provide connection parameters to database:')

        host = input('host (default 127.0.0.1): ') or '127.0.0.1'
        port = input('port (default 5432): ') or '5432'
        user = input('user (default postgres): ') or 'postgres'
        database = input('database name: ')
        password = input('password: ')

        conn = create_engine(
            'postgresql://' + '%s:%s@%s:%s/%s' % (user,
                                                  password,
                                                  host,
                                                  port,
                                                  database))

    Session = sessionmaker(bind=conn)
    session = Session()
    return session


def init_pypsa_network(time_range_lim):
    """
    Instantiate PyPSA network

    Parameters
    ----------
    time_range_lim:
    Returns
    -------
    network: PyPSA network object
        Contains powerflow problem
    snapshots: iterable
        Contains snapshots to be analyzed by powerplow calculation
    """
    network = pypsa.Network()
    network.set_snapshots(time_range_lim)
    snapshots = network.snapshots

    return network, snapshots


def get_pq_sets(session, table, scenario, columns=None, index_col=None):
    """

    Parameters
    ----------
    session: SQLAlchemy sessino object
    table: SQLAlchemy orm table object
        Specified pq-sets table
    scenario : str
        Name of chosen scenario
    columns: list of strings
        Columns to be selected from pq-sets table (default None)
    index_col: string
        Column to set index on (default None)

    Returns
    -------
    pq_set: pandas DataFrame
        Table with pq-Values to be applied in PF analysis
    """
 
        # retrieve table
    if columns is not None:
        pq_query = session.query(table).options(load_only(*columns)).\
            filter(table.scn_name==scenario)
    else:
        pq_query = session.query(table).filter(table.scn_name==scenario)
    pq_set = pd.read_sql_query(pq_query.statement,
                               session.bind,
                               index_col=index_col)
    # remove unwanted columns                           
    pq_set = pq_set.drop('temp_id',1)
    pq_set = pq_set.drop('scn_name',1)
    
    return pq_set


def get_timerange(session, temp_id_set, TempResolution, start_h=1, end_h=8760):
    """
    Parameters
    ----------
    session: SQLAlchemy session object
    temp_id_set : int
        ID of temporal coverage of power flow analysis
    TempResolution: SQLAlchemy orm class
        Table object of the table specifying temporal coverage of PFA
    start_h: int
        First hour of year used for calculations
    end_h: int
        Last hour of year used for calculations
    Returns
    -------
    timerange: Pandas DatetimeIndex
        Time range to be analyzed by PF
    """

    query = session.query(TempResolution.start_time).filter(
        TempResolution.temp_id == temp_id_set)
    start_date = query.all()
    start_date = ''.join(str(i) for i in start_date[0])

    query = session.query(TempResolution.timesteps).filter(
        TempResolution.temp_id == temp_id_set)
    periods = query.all()
    periods = int(''.join(str(i) for i in periods[0]))

    query = session.query(TempResolution.resolution).filter(
        TempResolution.temp_id == temp_id_set)
    frequency = query.all()
    frequency = ''.join(str(i) for i in frequency[0])

    timerange = pd.DatetimeIndex(freq=frequency,
                                 periods=periods,
                                 start=start_date)
    timerange = timerange[start_h-1:end_h]

    return timerange


def transform_timeseries4pypsa(timeseries, timerange, column=None):
    """
    Transform pq-set timeseries to PyPSA compatible format

    Parameters
    ----------
    timeseries: Pandas DataFrame
        Containing timeseries

    Returns
    -------
    pysa_timeseries: Pandas DataFrame
        Reformated pq-set timeseries
    """
    timeseries.index = [str(i) for i in timeseries.index]
    
    if column is None:
        pypsa_timeseries = timeseries.apply(
            pd.Series).transpose().set_index(timerange)
    else:
        pypsa_timeseries = timeseries[column].apply(
            pd.Series).transpose().set_index(timerange)

    return pypsa_timeseries


def import_components(tables, session, scenario):
    """
    Import PF power system components (Lines, Buses, Generators, ...)

    Parameters
    ----------
    tables: list of SQLAlchemy orm table object
        Considered power system component tables
    session : SQLAlchemy session object
        In this case it has to be a session connection to `OEDB`
    scenario : str
        Filter query by components of given scenario name

    Returns
    -------
    components: dict

    """
    component_data = {}

    for table in tables:
        if table.__name__ is not 'Transformer':
            id_col = str(table.__name__).lower() + "_id"
        elif table.__name__ is 'Transformer':
            id_col = 'trafo_id'
        if table.__name__ is not 'Source':
            query = session.query(table).filter(table.scn_name==scenario)
        elif table.__name__ is 'Source':
            query = session.query(table)            
        component_data[table.__name__] = pd.read_sql_query(
            query.statement, session.bind,
            index_col=id_col)

    return component_data

def create_powerflow_problem(timerange, components):
    """
    Create PyPSA network object and fill with data

    Parameters
    ----------
    timerange: Pandas DatetimeIndex
        Time range to be analyzed by PF
    components: dict

    Returns
    -------
    network: PyPSA powerflow problem object
    """

    # initialize powerflow problem
    network, snapshots = init_pypsa_network(timerange)
    

    # add components to network
    for component in components.keys():
        network.import_components_from_dataframe(components[component],
                                                 component)

    # add timeseries data

    return network, snapshots


def import_pq_sets(session, network, pq_tables, timerange, scenario, 
                   columns=None, start_h=1, end_h=8760):
    """
    Import pq-set series to PyPSA network

    Parameters
    ----------
    session : SQLAlchemy session object
        In this case it has to be a session connection to `OEDB`
    network : PyPSA network container
    pq_tables: Pandas DataFrame
        PQ set values for each time step
    scenario : str
        Filter query by pq-sets for components of given scenario name
    columns: list of strings
        Columns to be selected from pq-sets table (default None)
    start_h: int
        First hour of year used for calculations
    end_h: int
        Last hour of year used for calculations
        
    Returns
    -------
    network: PyPSA powerflow problem object
        Altered network object
    """
    
    for table in pq_tables:
        name = table.__table__.name.split('_')[0]
        index_col = name + '_id'
        component_name = name[:1].upper() + name[1:]
        pq_set = get_pq_sets(session, table, scenario, columns=columns,
                             index_col=index_col)
        if not pq_set.empty:
            for key in [x for x in ['p_set', 'q_set', 'v_mag_pu_set']
                        if x in pq_set.columns.values]:
                            
                # remove unneeded part of timeseries
                if start_h != 1 or end_h != 8760:
                    for idx in pq_set.index:
                        pq_set[key][idx] = pq_set[key][idx][start_h-1:end_h]
                
                series = transform_timeseries4pypsa(pq_set,
                                                    timerange,
                                                    column=key)
                io.import_series_from_dataframe(network,
                                                series,
                                                component_name,
                                                key)

    return network


def add_coordinates(network):
    """
    Add coordinates to nodes based on provided geom

    Parameters
    ----------
    network : PyPSA network container

    Returns
    -------
    Altered PyPSA network container ready for plotting
    """
    for idx, row in network.buses.iterrows():
        wkt_geom = to_shape(row['geom'])
        network.buses.loc[idx, 'x'] = wkt_geom.x
        network.buses.loc[idx, 'y'] = wkt_geom.y

    return network


def plot_line_loading(network, output='file'):
    """
    Plot line loading as color on lines

    Displays line loading relative to nominal capacity
    Parameters
    ----------
    network : PyPSA network container
        Holds topology of grid including results from powerflow analysis
    output : str
        Specify 'file' (saves to disk, default) or 'show' (show directly)
    """
    # TODO: replace p0 by max(p0,p1) and analogously for q0
    # TODO: implement for all given snapshots

    # calculate relative line loading as S/S_nom
    # with S = sqrt(P^2 + Q^2)
    loading = ((network.lines_t.p0.loc[network.snapshots[1]] ** 2 +
                network.lines_t.q0.loc[network.snapshots[1]] ** 2).apply(sqrt) \
               / (network.lines.s_nom * 1e3)) * 100  # to MW as results from pf

    # do the plotting
    ll = network.plot(line_colors=abs(loading), line_cmap=plt.cm.jet,
                      title="Line loading")

    # add colorbar, note mappable sliced from ll by [1]
    cb = plt.colorbar(ll[1])
    cb.set_label('Line loading in %')
    if output is 'show':
        plt.show()
    elif output is 'file':
        plt.savefig('Line_loading.png')


def add_source_types(session, network, table):
    """
    Get source table from OEDB, change source_id to source name
    
    Parameters
    ----------
    session : SQLAlchemy session object
        In this case it has to be a session connection to `OEDB`
    network : PyPSA network container
    tables: list of SQLAlchemy orm table object
        Considered power system component tables

    Returns
    -------
    None 
    """
    source = import_components(tables = [table], 
                               session = session, 
                               scenario = None)['Source']
    source = source.drop('commentary',1)
    
    network.generators = network.generators.drop('carrier',1).\
                         rename(columns={'source':'carrier'})
    
    for idx, row in network.generators.iterrows():
        source_name = source.loc[row['carrier'],'name']
        network.generators.loc[idx, 'carrier'] = source_name
        
    source = source.set_index(keys = source.name.values).drop('name',1)
    network.import_components_from_dataframe(source, 'Carrier')
    

def plot_stacked_gen(network, bus=None, resolution='GW'):
    """
    Plot stacked sum of generation grouped by carrier type
    
    
    Parameters
    ----------
    network : PyPSA network container
    bus: string
        Plot all generators at one specific bus. If none,
        sum is calulated for all buses
    resolution: string
        Unit for y-axis. Can be either GW/MW/KW

    Returns
    -------
    Plot 
    """
    if resolution == 'GW':
        reso_int = 1e3
    elif resolution == 'MW':
        reso_int = 1
    elif resolution == 'KW':
        reso_int = 0.001
        
    # sum for all buses
    if bus==None:    
        p_by_carrier = network.generators_t.p.\
                       groupby(network.generators.carrier, axis=1).sum()
    # sum for a single bus 
    elif bus is not None:
        filtered_gens = network.generators[network.generators['bus'] == bus]
        p_by_carrier = network.generators_t.p.\
                       groupby(filtered_gens.carrier, axis=1).sum()
        
    colors = {'biomass':'green', 
              'coal':'k', 
              'gas':'orange',
              'eeg_gas':'olive',
              'geothermal':'purple', 
              'lignite':'brown', 
              'oil':'darkgrey',
              'other_non_renewable':'pink', 
              'reservoir':'navy', 
              'run_of_river':'aqua',
              'pumped_storage':'steelblue', 
              'solar':'yellow', 
              'uranium':'lime',
              'waste':'sienna', 
              'wind':'skyblue'}
              
#    TODO: column reordering based on available columns

    fig,ax = plt.subplots(1,1)
    
    fig.set_size_inches(12,6)

    (p_by_carrier/reso_int).plot(kind="area",ax=ax,linewidth=4,
                            color=[colors[col] for col in p_by_carrier.columns])
    ax.legend(ncol=4,loc="upper left")
    
    ax.set_ylabel(resolution)
    ax.set_xlabel("")
    
    
if __name__ == '__main__':
    pass