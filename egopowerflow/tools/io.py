""" io.py

Input/output operations between powerflow schema in the oedb and PyPSA.
Additionally oedb wrapper classes to instantiate PyPSA network objects.


Attributes
----------

packagename: str
    Package containing orm class definitions
temp_ormclass: str
    Orm class name of table with temporal resolution
carr_ormclass: str
    Orm class name of table with carrier id to carrier name datasets

"""

__copyright__ = ""
__license__ = ""
__author__ = ""

import pypsa
from importlib import import_module
import pandas as pd
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import and_
from collections import OrderedDict
import re
import json
import os


packagename = 'egoio.db_tables'
temp_ormclass = 'TempResolution'
carr_ormclass = 'Source'

def loadcfg(path=''):
    if path == '':
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'config.json')
    return json.load(open(path), object_pairs_hook=OrderedDict)


class ScenarioBase():
    """ Base class to hide package/db handling
    """

    def __init__(self, session, method, version=None, *args, **kwargs):

        global temp_ormclass
        global carr_ormclass

        schema = 'model_draft' if version is None else 'grid'

        cfgpath = kwargs.get('cfgpath', '')
        self.config = loadcfg(cfgpath)[method]

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

        # map carrier id to carrier table
        self.map_ormclass(carr_ormclass)

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

    def __repr__(self):
        r = ('NetworkScenario: %s' % self.scn_name)

        if not self.network:
            r += "\nTo create a PyPSA network call <NetworkScenario>.build_network()."

        return r

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

    def id_to_source(self):

        ormclass = self._mapped['Source']
        query = self.session.query(ormclass)

        # TODO column naming in database
        return {k.source_id: k.name for k in query.all()}

    def by_scenario(self, name):
        """
        """

        ormclass = self._mapped[name]
        query = self.session.query(ormclass).filter(
            ormclass.scn_name == self.scn_name)

        if self.version:
            query = query.filter(ormclass.version == self.version)

        # TODO: Better handled in db
        if name == 'Transformer':
            name = 'Trafo'

        df = pd.read_sql(query.statement,
                         self.session.bind,
                         index_col=name.lower() + '_id')

        if 'source' in df:
            df.source = df.source.map(self.id_to_source())

        return df

    def series_by_scenario(self, name, column):
        """
        """

        ormclass = self._mapped[name]

        # TODO: pls make more robust
        id_column = re.findall(r'[A-Z][^A-Z]*', name)[0] + '_' + 'id'
        id_column = id_column.lower()

        query = self.session.query(
            getattr(ormclass, id_column),
            getattr(ormclass, column)[self.start_h: self.end_h].
            label(column)).filter(and_(
                ormclass.scn_name == self.scn_name,
                ormclass.temp_id == self.temp_id))

        if self.version:
            query = query.filter(ormclass.version == self.version)

        df = pd.io.sql.read_sql(query.statement,
                                self.session.bind,
                                columns=[column],
                                index_col=id_column)

        df.index = df.index.astype(str)

        # change of format to fit pypsa
        df = df[column].apply(pd.Series).transpose()

        try:
            assert not df.empty
            df.index = self.timeindex
        except AssertionError:
            print("No data for %s in column %s." % (name, column))

        return df

    def build_network(self, *args, **kwargs):
        """
        """
        # TODO: build_network takes care of divergences in database design and
        # future PyPSA changes from PyPSA's v0.6 on. This concept should be
        # replaced, when the oedb has a revision system in place, because
        # sometime this will break!!!

        network = pypsa.Network()
        network.set_snapshots(self.timeindex)

        timevarying_override = False

        if pypsa.__version__ == '0.8.0':

            old_to_new_name = {'Generator':
                               {'p_min_pu_fixed': 'p_min_pu',
                                'p_max_pu_fixed': 'p_max_pu',
                                'source': 'carrier',
                                'dispatch': 'former_dispatch'},
                               'Bus':
                               {'current_type': 'carrier'},
                               'Transformer':
                               {'trafo_id': 'transformer_id'},
                               'Storage':
                               {'p_min_pu_fixed': 'p_min_pu',
                                'p_max_pu_fixed': 'p_max_pu',
                                'soc_cyclic': 'cyclic_state_of_charge',
                                'soc_initial': 'state_of_charge_initial'}}

            timevarying_override = True

        else:

            old_to_new_name = {'Storage':
                               {'soc_cyclic': 'cyclic_state_of_charge',
                                'soc_initial': 'state_of_charge_initial'}}

        for comp, comp_t_dict in self.config.items():

            # TODO: This is confusing, should be fixed in db
            pypsa_comp_name = 'StorageUnit' if comp == 'Storage' else comp

            df = self.by_scenario(comp)

            if comp in old_to_new_name:

                tmp = old_to_new_name[comp]
                df.rename(columns=tmp, inplace=True)

            network.import_components_from_dataframe(df, pypsa_comp_name)

            if comp_t_dict:

                for comp_t, columns in comp_t_dict.items():

                    for col in columns:

                        df_series = self.series_by_scenario(comp_t, col)

                        # TODO: VMagPuSet?
                        if timevarying_override and comp == 'Generator':
                            idx = df[df.former_dispatch == 'flexible'].index
                            idx = [i for i in idx if i in df_series.columns]
                            df_series.drop(idx, axis=1, inplace=True)

                        try:

                            pypsa.io.import_series_from_dataframe(
                                network,
                                df_series,
                                pypsa_comp_name,
                                col)

                        except (ValueError, AttributeError):
                            print("Series %s of component %s could not be "
                                  "imported" % (col, pypsa_comp_name))

        self.network = network

        return network

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

if __name__ == '__main__':
    if pypsa.__version__ not in ['0.6.2', '0.8.0']:
        print('Pypsa version %s not supported.' % pypsa.__version__)
    pass
