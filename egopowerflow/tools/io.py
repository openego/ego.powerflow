""" DB wrapper for PyPSA...wip

Attributes
----------

packagename: str
    ...
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
import re
import os
import json


# for debugging
conn = engine('v0.2.10')
Session = sessionmaker(bind=conn)
session = Session()


packagename = 'egoio.db_tables'
configuration = json.load(open('config.json'), object_pairs_hook=OrderedDict)
temp_ormclass = 'TempResolution'
carr_ormclass = 'Source'

# TODO: ego.powerflow takes care of divergences in database design and future
# PyPSA changes from PyPSA's v0.6 on. This concept should be replaced, when the
# oedb has a revision system in place, because sometime this will break!!!
# def upgrade_structure(pypsa_version, mapped):

class ScenarioBase():
    """ Hide package/db stuff...
    """

    def __init__(self, session, method, version=None, *args, **kwargs):

        global configuration
        global temp_ormclass
        global carr_ormclass

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

        if self.network:
            r += "\nPyPSA network ready."

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


    def id_to_carrier(self):

        ormclass = self._mapped['Source']

        query = session.query(ormclass)

        # TODO column naming in database
        id_to_carrier = {k.source_id:k.name for k in query.all()}

        return id_to_carrier

    def by_scenario(self, name):
        """
        """

        ormclass = self._mapped[name]

        query = session.query(ormclass).filter(ormclass.scn_name == self.scn_name)

        if self.version:
            query = query.filter(ormclass.version == self.version)

        # TODO: Better handled in db
        if name == 'Transformer':
            name = 'Trafo'

        df = pd.read_sql(query.statement,
                           session.bind,
                           index_col=name.lower() + '_id')

        if 'source' in df:
            df.source = df.source.map(self.id_to_carrier())

        return df

    def series_by_scenario(self, name, column):
        """
        """

        ormclass = self._mapped[name]

        # TODO: pls make more robust
        id_column = re.findall(r'[A-Z][^A-Z]*', name)[0] + '_' + 'id'
        id_column = id_column.lower()

        query = session.query(
            getattr(ormclass,id_column),
            getattr(ormclass,column)[self.start_h:self.end_h].\
            label(column)).filter(and_(
            ormclass.scn_name == self.scn_name,
            ormclass.temp_id == self.temp_id))

        if self.version:
            query = query.filter(ormclass.version == self.version)

        try:
            df =  pd.io.sql.read_sql(query.statement,
                            session.bind,
                            columns=[column],
                            index_col=id_column)

            df.index = df.index.astype(str)

            # change of format to fit pypsa
            df = df[column].apply(pd.Series).transpose()

            df.index = self.timeindex

            assert not df.empty
            return df

        except (ValueError, AssertionError):
            print("No data for %s in column %s." % (name, column))


    def build_network(self, *args, **kwargs):
        """
        """

        network = pypsa.Network()
        network.set_snapshots(self.timeindex)


        for c, v in self.config.items():

            # TODO: This is confusing, should be fixed in db
            name = 'StorageUnit' if c == 'Storage' else c

            network.import_components_from_dataframe(self.by_scenario(c), name)

            if isinstance(v, dict):
                for cc, vv in v.items():
                    for col in vv:
                        df = self.series_by_scenario(cc, col)
                        try:
                            pypsa.io.import_series_from_dataframe(network, df, name, col)
                        except (ValueError, AttributeError):
                            print("Series %s of component %s could not be "
                                  "imported" % (col, name))

        self.network = network

        return network

# for debugging
from egopowerflow.tools.plot import plot_line_loading, add_coordinates

md = NetworkScenario(session, method='lopf', end_h=2, start_h=1,
                          scn_name='Status Quo')
mdnw = md.build_network()
mdnw.lopf(snapshots=md.timeindex, solver_name='gurobi')
mdnw = add_coordinates(mdnw)
plot_line_loading(mdnw)

mdpf = NetworkScenario(session, method='pf', end_h=2, start_h=1,
                          scn_name='Status Quo')
mdpfnw = mdpf.build_network()
mdpfnw.pf(snapshots=mdpf.timeindex)

gr = NetworkScenario(session, method='lopf', end_h=2, start_h=1, version='v0.2.10',
                     prefix='EgoPfHv', scn_name='Status Quo')
grnw = gr.build_network()
grnw.lopf(snapshots=gr.timeindex, solver_name='gurobi')
grnw = add_coordinates(grnw)
plot_line_loading(grnw)

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
