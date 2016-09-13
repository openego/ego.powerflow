import pypsa
import pandas as pd

from sqlalchemy.orm import sessionmaker, load_only
from sqlalchemy import create_engine
from datetime import datetime
from pypsa import io
from oemof import db
from geoalchemy2.shape import to_shape

from egoio.db_tables.calc_ego_mv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution, BusVMagSet, GeneratorPqSet, LoadPqSet

def oedb_session(section='oedb'):
    """Get SQLAlchemy session object with valid connection to OEDB"""

    # get session object by oemof.db tools (requires .oemof/config.ini
    try:
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


def get_pq_sets(session, table, columns=None, index_col=None, slicer=None):
    """

    Parameters
    ----------
    session: SQLAlchemy sessino object
    table: SQLAlchemy orm table object
        Specified pq-sets table
    columns: list of strings
        Columns to be selected from pq-sets table (default None)
    index_col: string
        Column to set index on (default None)
    slicer: list of int's
        Slices array of time-dependent p/q-values to apply in PF (default None)

    Returns
    -------
    pq_set: pandas DataFrame
        Table with pq-Values to be applied in PF analysis
    """

    # retrieve table
    if columns is not None:
        pq_query = session.query(table).options(load_only(*columns))
    else:
        pq_query = session.query(table)
    pq_set = pd.read_sql_query(pq_query.statement,
                               session.bind,
                               index_col=index_col)

    # slice relevant part by given slicer
    #TODO: implement slicing of p,q-array

    return pq_set


def get_timerange(session, temp_id_set):
    """
    Parameters
    ----------
    session: SQLAlchemy session object

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

    if column is None:
        pypsa_timeseries = timeseries.apply(
            pd.Series).transpose().set_index(timerange)
    else:
        pypsa_timeseries = timeseries[column].apply(
            pd.Series).transpose().set_index(timerange)

    return pypsa_timeseries


def import_components(tables):
    """
    Import PF power system components (Lines, Buses, Generators, ...)

    Parameters
    ----------
    tables: list of SQLAlchemy orm table object
        Considered power system component tables

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

        component_data[table.__name__] = pd.read_sql_query(
            session.query(table).statement, session.bind,
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
    network

    # add components to network
    for component in components.keys():
        network.import_components_from_dataframe(components[component],
                                                 component)

    # add timeseries data

    return network, snapshots


def import_pq_sets(session, network, pq_tables, timerange):
    """
    Import pq-set series to PyPSA network

    Parameters
    ----------
    pq_tables: Pandas DataFrame
        PQ set values for each time step

    Returns
    -------
    network: PyPSA powerflow problem object
        Altered network object
    """

    for table in pq_tables:
        name = table.__table__.name.split('_')[0]
        index_col = name + '_id'
        component_name = name[:1].upper() + name[1:]
        pq_set = get_pq_sets(session, table,
                             index_col=index_col)
        if not pq_set.empty:
            for key in [x for x in ['p_set', 'q_set', 'v_mag_pu_set']
                        if x in pq_set.columns.values]:
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
if __name__ == '__main__':
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
    components = import_components(tables)

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

    # close session
    session.close()