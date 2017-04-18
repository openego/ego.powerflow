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


import pypsa
import pandas as pd

from pypsa import io
from numpy import isnan
from collections import OrderedDict
from sqlalchemy.inspection import inspect


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
    pypsa_timeseries: Pandas DataFrame
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

    component_data = OrderedDict()

    for table in tables:

        name = table.__name__.split('Hv')[-1]
        id_col = name.lower() + '_id'

        if table.__name__ is not 'EgoGridPfHvSource':
            query = session.query(table).filter(table.scn_name == scenario)

        elif table.__name__ is 'EgoGridPfHvSource':
            query = session.query(table)

        if table.__name__ is 'EgoGridPfHvStorage':
            name = 'StorageUnit'

        component_data[name] = pd.read_sql_query(
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

    # fix names of certain columns for storage units
    if 'EgoGridPfHvStorage' in components.keys():
        fix_storages(network)

    return network, snapshots


def get_pq_sets(session, table, scenario, start_h, end_h, column=None,
                index_col=None):
    """
    Parameters
    ----------
    session: SQLAlchemy sessino object
    table: SQLAlchemy orm table object
        Specified pq-sets table
    scenario : str
        Name of chosen scenario
    start_h: int
        First hour of year used for calculations
    end_h: int
        Last hour of year used for calculations
    columns: list of strings
        Columns to be selected from pq-sets table (default None)
    index_col: string
        Column to set index on (default None)

    Returns
    -------
    pq_set: pandas DataFrame
        Table with pq-Values to be applied in PF analysis
    """

    id_column = inspect(table).primary_key[1].name

    try:
        assert 'id' in id_column
    except AssertionError:
        print('There is no id column in %s', table)

    query = session.query(getattr(table, id_column),
                          getattr(table, column)[start_h:end_h])
    query = query.filter(table.scn_name == scenario)

    pq_set = pd.read_sql(query.statement,
                         session.bind,
                         index_col=index_col)

    pq_set.columns = [column]
    return pq_set


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
        name = table.__table__.name.split('_')[4]
        index_col = name + '_id'
        component_name = name[:1].upper() + name[1:]
        if table.__name__ is 'EgoGridPfHvStorage':
            index_col = 'storage_id'
            component_name = 'StorageUnit'

        for column in columns:

            pq_set = get_pq_sets(session,
                                 table,
                                 scenario,
                                 column=column,
                                 index_col=index_col,
                                 start_h=start_h,
                                 end_h=end_h)

            series = transform_timeseries4pypsa(pq_set,
                                                timerange,
                                                column=column)
            if column is 'soc_set':
                column = 'state_of_charge_set'
            io.import_series_from_dataframe(network,
                                            series,
                                            component_name,
                                            column)

    return network


def add_source_types(session, network, table):
    """
    Get source table from OEDB, change source_id to source name

    Parameters
    ----------
    session : SQLAlchemy session object
        In this case it has to be a session connection to `OEDB`
    network : PyPSA network container
    table:  SQLAlchemy orm table object ("Source" table)
        Considered power system component tables

    Returns
    -------
    None
    """

    source = import_components(tables=[table],
                               session=session,
                               scenario=None)['Source']
    source = source.drop('commentary', 1)

    carrier = network.generators.carrier.astype('int')

    carrier = carrier.map(source.name)

    carrier.fillna('unknown', inplace=True)

    network.generators.carrier = carrier

#     network.generators = network.generators.drop('carrier',1).\
#                          rename(columns={'source':'carrier'})
#
#
#     for idx, row in network.generators.iterrows():
#         if isnan(network.generators.loc[idx].carrier):
#             network.generators.loc[idx, 'carrier'] = 'unknown'
#         else:
#             source_name = source.loc[row['carrier'],'name']
#             network.generators.loc[idx, 'carrier'] = source_name
#
    source = source.set_index(keys=source.name.values).drop('name',1)

    network.import_components_from_dataframe(source, 'Carrier')


def results_to_oedb(session, network, grid='mv'):
    """Return results obtained from PyPSA to oedb"""
    # moved this here to prevent error when not using the mv-schema
    if grid.lower() == 'mv':
        from egoio.db_tables.calc_ego_mv_powerflow import ResBus, ResLine, ResTransformer
    elif grid.lower() == 'hv':
        print('Not implemented: Result schema for HV missing')
    else:
        print('Please enter mv or hv!')
    # from oemof import db
    # engine = db.engine(section='oedb')
    # from egoio.db_tables import calc_ego_mv_powerflow
    # calc_ego_mv_powerflow.Base.metadata.create_all(engine)

    #TODO: make this more safe, at least inform the user about the deleting results
    # empty all results table
    session.query(ResBus).delete()
    session.query(ResLine).delete()
    session.query(ResTransformer).delete()
    session.commit()

    # insert voltage at buses to database
    for col in network.buses_t.v_mag_pu:
        res_bus = ResBus(
            bus_id=col,
            v_mag_pu=network.buses_t.v_mag_pu[col].tolist()
        )
        session.add(res_bus)
    session.commit()

    # insert active and reactive power of lines to database
    for col in network.lines_t.p0:
        res_line = ResLine(
            line_id=col,
            p0=network.lines_t.p0[col].tolist(),
            q0=network.lines_t.q0[col].tolist(),
            p1=network.lines_t.p1[col].tolist(),
            q1=network.lines_t.q1[col].tolist()
        )
        session.add(res_line)
    session.commit()

    # insert active and reactive power of lines to database
    for col in network.transformers_t.p0:
        res_transformer = ResLine(
            trafo_id=col,
            p0=network.transformers_t.p0[col].tolist(),
            q0=network.transformers_t.q0[col].tolist(),
            p1=network.transformers_t.p1[col].tolist(),
            q1=network.transformers_t.q1[col].tolist()
        )
        session.add(res_transformer)
    session.commit()


def fix_storages(network):
    """
    Workaround to deal with the new name for storages
    used by PyPSA.
    Old: Storage
    New: StorageUnit

    Parameters
    ----------
    network : PyPSA network container

    Returns
    -------
    None
    """
    network.storage_units = network.storage_units.drop('state_of_charge_initial',1).\
                     rename(columns={'soc_initial':'state_of_charge_initial'})
    network.storage_units = network.storage_units.drop('cyclic_state_of_charge',1).\
                     rename(columns={'soc_cyclic':'cyclic_state_of_charge'})

if __name__ == '__main__':
    pass
