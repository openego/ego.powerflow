""" DB wrapper for PyPSA...wip
"""

__copyright__ = ""
__license__ = ""
__author__ = ""

import pypsa
from sqlalchemy.orm import sessionmaker
from oemof.db import engine
from importlib import import_module
import pandas as pd
from sqlalchemy import or_, and_, exists, inspect
from sqlalchemy.orm.exc import NoResultFound
from collections import OrderedDict
import os


# debug
conn = engine('open_ego')
Session = sessionmaker(bind=conn)
session = Session()


packagename = 'egoio.db_tables'
# TODO: Maybe there is a better container desigen???
configuration = {'lopf':
                {'Bus': None,
                'Generator': {'GeneratorPqSet': ['p_set', 'p_max_pu']},
                'Line': None,
                'Transformer': None,
                'Load': {'LoadPqSet': ['p_set', 'q_set']},
                'Storage': {'StoragePqSet': ['p_set']}}}

temp_ormclass = 'TempResolution'


class ScenarioBase():
    """ Hide package/db stuff...
    """

    def __init__(self, session, method, version=None, *args, **kwargs):

        global configuration
        global temp_ormclass

        schema = 'model_draft' if version is None else 'grid'

        self.config = configuration[method].copy()
        self.session = session
        self.version = version
        self._prefix = kwargs.get('prefix', 'EgoGridPfHv')
        self._pkg = import_module(packagename + '.' + schema)
        self._mapped = {}

        # map static and timevarying classes
        for k, v in self.config.items():
            self.map_ormclass(k)
            if isinstance(v, dict):
                for kk in v.keys():
                    self.map_ormclass(kk)

        # map temporal resolution table
        self.map_ormclass(temp_ormclass)

    def map_ormclass(self, name):

        global packagename

        try:
            self._mapped[name] = getattr(self._pkg, self._prefix + name)

        except AttributeError:
            print('Warning: Relation %s does not exist.' % name)


class NetworkScenario(ScenarioBase):
    """
    """

    def __init__(self, session, *args, **kwargs):
        super().__init__(session, *args, **kwargs)

        self.scn_name = kwargs.get('scn_name', 'Status Quo')
        self.method   = kwargs.get('method', 'lopf')
        self.start_h  = kwargs.get('start_h', 1)
        self.end_h    = kwargs.get('end_h', 20)
        self.temp_id  = kwargs.get('temp_id', 1)
        self.network  = None

        self.configure_timeindex()

    def configure_timeindex(self):
        """
        """

        try:

            ormclass = self._mapped['TempResolution']
            tr = self.session.query(ormclass).filter(
                ormclass.temp_id == self.temp_id).one()

        except (KeyError, NoResultFound):
            print('temp_id %s does not exist.' % self.temp_id)

        timeindex = pd.DatetimeIndex(start=tr.start_time,
                                     periods=tr.timesteps,
                                     freq=tr.resolution)

        self.timeindex = timeindex[self.start_h - 1: self.end_h]


    def by_scenario(self, name):
        """
        """

        ormclass = self._mapped[name]

        query = session.query(ormclass).filter(ormclass.scn_name == self.scn_name)

        if self.version:
            query = query.filter(ormclass.version == self.version)

        return pd.read_sql(query.statement,
                           session.bind,
                           index_col=name.lower() + '_id')


    def series_by_scenario(self, name, column):
        """
        """

        ormclass = self._mapped[name]

        # TODO: Make more robust
        id_column = inspect(ormclass).primary_key[1].name

        try:
            assert 'id' in id_column
        except AssertionError:
            print('There is no id column in %s', ormclass)

        query = session.query(
            getattr(ormclass,id_column),
            getattr(ormclass,column)).filter(and_(
            ormclass.scn_name == self.scn_name,
            ormclass.temp_id == self.temp_id))

        if self.version:
            query = query.filter(ormclass.version == self.version)

        return pd.read_sql(query.statement,
                           session.bind,
                           index_col=id_column)


    def build_network(self):
        """
        """

        network = pypsa.Network(snapshots=self.timeindex)

        network = pypsa.Network()

        for c, v in self.config.items():
            # TODO: This should be managed in the DATABASE itself
            if c == 'Storage':
                network.import_components_from_dataframe(scenario.by_scenario(c),'StorageUnit')
            else:
                network.import_components_from_dataframe(scenario.by_scenario(c),c)

            if isinstance(v, dict):
                for cc, vv in v.items():
                    for col in vv:
                        pass


scenario = NetworkScenario(session, method='lopf', end_h=10, start_h=1,
                          scn_name='Status Quo')
scenario.build_network()


# from egoio.db_tables.model_draft import EgoGridPfHvBus as Bus, EgoGridPfHvLine as Line, EgoGridPfHvGenerator as Generator, EgoGridPfHvLoad as Load,\
#     EgoGridPfHvTransformer as Transformer, EgoGridPfHvTempResolution as TempResolution, EgoGridPfHvGeneratorPqSet as GeneratorPqSet,\
#     EgoGridPfHvLoadPqSet as LoadPqSet, EgoGridPfHvSource as Source #, EgoGridPfHvStorage,\
#
#
#
# import pypsa
# import pandas as pd
#
# from pypsa import io
# from numpy import isnan
# from collections import OrderedDict
# from sqlalchemy.inspection import inspect
#
#
# def init_pypsa_network(time_range_lim):
#     """
#     Instantiate PyPSA network
#
#     Parameters
#     ----------
#     time_range_lim:
#     Returns
#     -------
#     network: PyPSA network object
#         Contains powerflow problem
#     snapshots: iterable
#         Contains snapshots to be analyzed by powerplow calculation
#     """
#     network = pypsa.Network()
#     network.set_snapshots(time_range_lim)
#     snapshots = network.snapshots
#
#     return network, snapshots
#
#
# def get_timerange(session, temp_id_set, TempResolution, start_h=1, end_h=8760):
#     """
#     Parameters
#     ----------
#     session: SQLAlchemy session object
#     temp_id_set : int
#         ID of temporal coverage of power flow analysis
#     TempResolution: SQLAlchemy orm class
#         Table object of the table specifying temporal coverage of PFA
#     start_h: int
#         First hggour of year used for calculations
#     end_h: int
#         Last hour of year used for calculations
#     Returns
#     -------
#     timerange: Pandas DatetimeIndex
#         Time range to be analyzed by PF
#     """
#
#     query = session.query(TempResolution.start_time).filter(
#         TempResolution.temp_id == temp_id_set)
#     start_date = query.all()
#     start_date = ''.join(str(i) for i in start_date[0])
#
#     query = session.query(TempResolution.timesteps).filter(
#         TempResolution.temp_id == temp_id_set)
#     periods = query.all()
#
#     periods = int(''.join(str(i) for i in periods[0]))
#
#     query = session.query(TempResolution.resolution).filter(
#         TempResolution.temp_id == temp_id_set)
#     frequency = query.all()
#     frequency = ''.join(str(i) for i in frequency[0])
#
#     timerange = pd.DatetimeIndex(freq=frequency,
#                                  periods=periods,
#                                  start=start_date)
#     timerange = timerange[start_h-1:end_h]
#
#     return timerange
#
#
#def transform_timeseries4pypsa(timeseries, timerange, column=None):
#    """
#    Transform pq-set timeseries to PyPSA compatible format
#
#    Parameters
#    ----------
#    timeseries: Pandas DataFrame
#        Containing timeseries
#
#    Returns
#    -------
#    pypsa_timeseries: Pandas DataFrame
#        Reformated pq-set timeseries
#    """
#    timeseries.index = [str(i) for i in timeseries.index]
#
#    if column is None:
#        pypsa_timeseries = timeseries.apply(
#            pd.Series).transpose().set_index(timerange)
#    else:
#        pypsa_timeseries = timeseries[column].apply(
#            pd.Series).transpose().set_index(timerange)
#
#    return pypsa_timeseries
#
#
#
#
#def import_components(tables, session, scenario):
#    """
#    Import PF power system components (Lines, Buses, Generators, ...)
#
#    Parameters
#    ----------
#    tables: list of SQLAlchemy orm table object
#        Considered power system component tables
#    session : SQLAlchemy session object
#        In this case it has to be a session connection to `OEDB`
#    scenario : str
#        Filter query by components of given scenario name
#
#    Returns
#    -------
#    components: dict
#
#    """
#
#    component_data = OrderedDict()
#
#    for table in tables:
#
#        name = table.__name__.split('Hv')[-1]
#        id_col = name.lower() + '_id'
#
#        if table.__name__ is not 'EgoGridPfHvSource':
#            query = session.query(table).filter(table.scn_name == scenario)
#
#        elif table.__name__ is 'EgoGridPfHvSource':
#            query = session.query(table)
#
#        if table.__name__ is 'EgoGridPfHvStorage':
#            name = 'StorageUnit'
#
#        component_data[name] = pd.read_sql_query(
#            query.statement, session.bind,
#            index_col=id_col)
#
#    return component_data
#
#
#def create_powerflow_problem(timerange, components):
#    """
#    Create PyPSA network object and fill with data
#
#    Parameters
#    ----------
#    timerange: Pandas DatetimeIndex
#        Time range to be analyzed by PF
#    components: dict
#
#    Returns
#    -------
#    network: PyPSA powerflow problem object
#    """
#
#    # initialize powerflow problem
#    network, snapshots = init_pypsa_network(timerange)
#
#    # add components to network
#    for component in components.keys():
#        network.import_components_from_dataframe(components[component],
#                                                 component)
#
#    # fix names of certain columns for storage units
#    if 'EgoGridPfHvStorage' in components.keys():
#        fix_storages(network)
#
#    return network, snapshots
#
#
#def get_pq_sets(session, table, scenario, start_h, end_h, column=None,
#                index_col=None):
#    """
#    Parameters
#    ----------
#    session: SQLAlchemy sessino object
#    table: SQLAlchemy orm table object
#        Specified pq-sets table
#    scenario : str
#        Name of chosen scenario
#    start_h: int
#        First hour of year used for calculations
#    end_h: int
#        Last hour of year used for calculations
#    columns: list of strings
#        Columns to be selected from pq-sets table (default None)
#    index_col: string
#        Column to set index on (default None)
#
#    Returns
#    -------
#    pq_set: pandas DataFrame
#        Table with pq-Values to be applied in PF analysis
#    """
#
#    id_column = inspect(table).primary_key[1].name
#
#    try:
#        assert 'id' in id_column
#    except AssertionError:
#        print('There is no id column in %s', table)
#
#    query = session.query(getattr(table, id_column),
#                          getattr(table, column)[start_h:end_h])
#    query = query.filter(table.scn_name == scenario)
#
#    pq_set = pd.read_sql(query.statement,
#                         session.bind,
#                         index_col=index_col)
#
#    pq_set.columns = [column]
#    return pq_set
#
#
#def import_pq_sets(session, network, pq_tables, timerange, scenario,
#                   columns=None, start_h=1, end_h=8760):
#    """
#    Import pq-set series to PyPSA network
#
#    Parameters
#    ----------
#    session : SQLAlchemy session object
#        In this case it has to be a session connection to `OEDB`
#    network : PyPSA network container
#    pq_tables: Pandas DataFrame
#        PQ set values for each time step
#    scenario : str
#        Filter query by pq-sets for components of given scenario name
#    columns: list of strings
#        Columns to be selected from pq-sets table (default None)
#    start_h: int
#        First hour of year used for calculations
#    end_h: int
#        Last hour of year used for calculations
#
#    Returns
#    -------
#    network: PyPSA powerflow problem object
#        Altered network object
#    """
#
#    for table in pq_tables:
#        name = table.__table__.name.split('_')[4]
#        index_col = name + '_id'
#        component_name = name[:1].upper() + name[1:]
#        if table.__name__ is 'EgoGridPfHvStorage':
#            index_col = 'storage_id'
#            component_name = 'StorageUnit'
#
#        for column in columns:
#
#            pq_set = get_pq_sets(session,
#                                 table,
#                                 scenario,
#                                 column=column,
#                                 index_col=index_col,
#                                 start_h=start_h,
#                                 end_h=end_h)
#
#            series = transform_timeseries4pypsa(pq_set,
#                                                timerange,
#                                                column=column)
#            if column is 'soc_set':
#                column = 'state_of_charge_set'
#            io.import_series_from_dataframe(network,
#                                            series,
#                                            component_name,
#                                            column)
#
#    return network
#
#
#def add_source_types(session, network, table):
#    """
#    Get source table from OEDB, change source_id to source name
#
#    Parameters
#    ----------
#    session : SQLAlchemy session object
#        In this case it has to be a session connection to `OEDB`
#    network : PyPSA network container
#    table:  SQLAlchemy orm table object ("Source" table)
#        Considered power system component tables
#
#    Returns
#    -------
#    None
#    """
#
#    source = import_components(tables=[table],
#                               session=session,
#                               scenario=None)['Source']
#    source = source.drop('commentary', 1)
#
#    carrier = network.generators.carrier.astype('int')
#
#    carrier = carrier.map(source.name)
#
#    carrier.fillna('unknown', inplace=True)
#
#    network.generators.carrier = carrier
#
##     network.generators = network.generators.drop('carrier',1).\
##                          rename(columns={'source':'carrier'})
##
##
##     for idx, row in network.generators.iterrows():
##         if isnan(network.generators.loc[idx].carrier):
##             network.generators.loc[idx, 'carrier'] = 'unknown'
##         else:
##             source_name = source.loc[row['carrier'],'name']
##             network.generators.loc[idx, 'carrier'] = source_name
##
#    source = source.set_index(keys=source.name.values).drop('name',1)
#
#    network.import_components_from_dataframe(source, 'Carrier')
#
#
#def results_to_oedb(session, network, grid='mv'):
#    """Return results obtained from PyPSA to oedb"""
#    # moved this here to prevent error when not using the mv-schema
#    if grid.lower() == 'mv':
#        from egoio.db_tables.calc_ego_mv_powerflow import ResBus, ResLine, ResTransformer
#    elif grid.lower() == 'hv':
#        print('Not implemented: Result schema for HV missing')
#    else:
#        print('Please enter mv or hv!')
#    # from oemof import db
#    # engine = db.engine(section='oedb')
#    # from egoio.db_tables import calc_ego_mv_powerflow
#    # calc_ego_mv_powerflow.Base.metadata.create_all(engine)
#
#    #TODO: make this more safe, at least inform the user about the deleting results
#    # empty all results table
#    session.query(ResBus).delete()
#    session.query(ResLine).delete()
#    session.query(ResTransformer).delete()
#    session.commit()
#
#    # insert voltage at buses to database
#    for col in network.buses_t.v_mag_pu:
#        res_bus = ResBus(
#            bus_id=col,
#            v_mag_pu=network.buses_t.v_mag_pu[col].tolist()
#        )
#        session.add(res_bus)
#    session.commit()
#
#    # insert active and reactive power of lines to database
#    for col in network.lines_t.p0:
#        res_line = ResLine(
#            line_id=col,
#            p0=network.lines_t.p0[col].tolist(),
#            q0=network.lines_t.q0[col].tolist(),
#            p1=network.lines_t.p1[col].tolist(),
#            q1=network.lines_t.q1[col].tolist()
#        )
#        session.add(res_line)
#    session.commit()
#
#    # insert active and reactive power of lines to database
#    for col in network.transformers_t.p0:
#        res_transformer = ResLine(
#            trafo_id=col,
#            p0=network.transformers_t.p0[col].tolist(),
#            q0=network.transformers_t.q0[col].tolist(),
#            p1=network.transformers_t.p1[col].tolist(),
#            q1=network.transformers_t.q1[col].tolist()
#        )
#        session.add(res_transformer)
#    session.commit()
#
#
#def fix_storages(network):
#    """
#    Workaround to deal with the new name for storages
#    used by PyPSA.
#    Old: Storage
#    New: StorageUnit
#
#    Parameters
#    ----------
#    network : PyPSA network container
#
#    Returns
#    -------
#    None
#    """
#    network.storage_units = network.storage_units.drop('state_of_charge_initial',1).\
#                     rename(columns={'soc_initial':'state_of_charge_initial'})
#    network.storage_units = network.storage_units.drop('cyclic_state_of_charge',1).\
#                     rename(columns={'soc_cyclic':'cyclic_state_of_charge'})
#
#if __name__ == '__main__':
#    pass
