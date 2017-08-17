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
from sqlalchemy import and_, func
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
    
def clear_results_db(session):
    from egoio.db_tables.model_draft import EgoGridPfHvResultBus as BusResult,\
                                                EgoGridPfHvResultStorage as StorageResult,\
                                                EgoGridPfHvResultGenerator as GeneratorResult,\
                                                EgoGridPfHvResultLine as LineResult,\
                                                EgoGridPfHvResultTransformer as TransformerResult,\
                                                EgoGridPfHvResultMeta as ResultMeta
    session.query(BusResult).delete()
    session.query(StorageResult).delete()
    session.query(GeneratorResult).delete()
    session.query(LineResult).delete()
    session.query(TransformerResult).delete()
    session.query(ResultMeta).delete()
    session.commit()


def results_to_oedb(session, network, grid, args):
    """Return results obtained from PyPSA to oedb"""
    # moved this here to prevent error when not using the mv-schema
    import datetime
    if grid.lower() == 'mv':
        print('MV currently not implemented')
    elif grid.lower() == 'hv':
        from egoio.db_tables.model_draft import EgoGridPfHvResultBus as BusResult,\
                                                EgoGridPfHvResultStorage as StorageResult,\
                                                EgoGridPfHvResultGenerator as GeneratorResult,\
                                                EgoGridPfHvResultLine as LineResult,\
                                                EgoGridPfHvResultTransformer as TransformerResult,\
                                                EgoGridPfHvResultMeta as ResultMeta
    else:
        print('Please enter mv or hv!')

    # get last result id and get new one
    last_res_id = session.query(func.max(ResultMeta.result_id)).scalar()
    if last_res_id == None:
        new_res_id = 1
    else: 
        new_res_id = last_res_id + 1
    
    # result meta data    
    res_meta = ResultMeta(
            result_id=new_res_id,
            scn_name=args['scn_name'],
            calc_date= datetime.datetime.now(),
            method=args['method'],
            network_clustering = args['network_clustering'],
            gridversion = args['gridversion'],
            start_h = args['start_h'],
            end_h = args['end_h'],
            solver = args['solver'],
            branch_cap_factor = args['branch_capacity_factor'],
            storage_extendable = args['storage_extendable'],
            load_shedding = args['load_shedding'],
            generator_noise = args['generator_noise'],
            commentary=args['comments']
    )
    session.add(res_meta)
    session.commit()
    
    # bus results
    try:
        for col in network.buses_t.v_mag_pu:
            res_bus = BusResult(
                result_id=new_res_id,
                bus_id=col,
                v_nom=network.buses.v_nom[col],
                current_type=network.buses.carrier[col],
                v_mag_pu_min = network.buses.v_mag_pu_min[col],
                v_mag_pu_max = network.buses.v_mag_pu_max[col],
                p=network.buses_t.p[col].tolist(),
                q=network.buses_t.q[col].tolist(),
                v_mag_pu=network.buses_t.v_mag_pu[col].tolist(),
                v_ang=network.buses_t.v_ang[col].tolist(),
                marginal_price=network.buses_t.marginal_price[col].tolist(),
                geom = network.buses.geom[col]
            )
            session.add(res_bus)
    except:
        for col in network.buses_t.v_mag_pu:
            res_bus = BusResult(
                result_id=new_res_id,
                bus_id=col,
                v_nom=network.buses.v_nom[col],
                current_type=network.buses.carrier[col],
                v_mag_pu_min = network.buses.v_mag_pu_min[col],
                p=network.buses_t.p[col].tolist(),
                v_mag_pu=network.buses_t.v_mag_pu[col].tolist(),
                v_ang=network.buses_t.v_ang[col].tolist(),
                marginal_price=network.buses_t.marginal_price[col].tolist(),
                geom = network.buses.geom[col]
            )
            session.add(res_bus)
    session.commit()

    # generator results
    try:
        for col in network.generators_t.p:
            res_gen = GeneratorResult(
                result_id=new_res_id,
                generator_id=col,
                bus=int(network.generators.bus[col]),
                dispatch=network.generators.former_dispatch[col],
                control=network.generators.control[col],
                p_nom=network.generators.p_nom[col],
                p_nom_extendable=network.generators.p_nom_extendable[col],
                p_nom_min=network.generators.p_nom_min[col],
                p_nom_max=network.generators.p_nom_max[col],
                p_min_pu_fixed=network.generators.p_min_pu[col],
                p_max_pu_fixed=network.generators.p_max_pu[col],
                sign=network.generators.sign[col],
#                source=network.generators.carrier[col],
                marginal_cost=network.generators.marginal_cost[col],
                capital_cost=network.generators.capital_cost[col],
                efficiency=network.generators.efficiency[col],
                p=network.generators_t.p[col].tolist(),
                q=network.generators_t.q[col].tolist(),
                p_nom_opt=network.generators.p_nom_opt[col]
            )
            session.add(res_gen)
    except:
        for col in network.generators_t.p:
            res_gen = GeneratorResult(
                result_id=new_res_id,
                generator_id=col,
                bus=int(network.generators.bus[col]),
                dispatch=network.generators.former_dispatch[col],
                control=network.generators.control[col],
                p_nom=network.generators.p_nom[col],
                p_nom_extendable=bool(network.generators.p_nom_extendable[col]),
                p_nom_min=network.generators.p_nom_min[col],
                p_nom_max=network.generators.p_nom_max[col],
                p_min_pu_fixed=network.generators.p_min_pu[col],
                p_max_pu_fixed=network.generators.p_max_pu[col],
                sign=network.generators.sign[col],
#                source=network.generators.carrier[col],
                marginal_cost=network.generators.marginal_cost[col],
                capital_cost=network.generators.capital_cost[col],
                efficiency=network.generators.efficiency[col],
                p=network.generators_t.p[col].tolist(),
                p_nom_opt=network.generators.p_nom_opt[col]
            )
            session.add(res_gen)
    session.commit()

    # line results
    try:
        for col in network.lines_t.p0:
            res_line = LineResult(
                result_id=new_res_id, 
                line_id=col,
                bus0=int(network.lines.bus0[col]),
                bus1=int(network.lines.bus1[col]),
                x=network.lines.x[col],
                r=network.lines.r[col],
                g=network.lines.g[col],
                b=network.lines.b[col],
                s_nom=network.lines.s_nom[col],
                s_nom_extendable=bool(network.lines.s_nom_extendable[col]),
                s_nom_min=network.lines.s_nom_min[col],
                s_nom_max=network.lines.s_nom_max[col],
                capital_cost=network.lines.capital_cost[col],
                length=network.lines.length[col],
                cables=int(network.lines.cables[col]),
                frequency=network.lines.frequency[col],
#                terrain_factor=network.lines.terrain_factor[col],
                p0=network.lines_t.p0[col].tolist(),
                q0=network.lines_t.q0[col].tolist(),
                p1=network.lines_t.p1[col].tolist(),
                q1=network.lines_t.q1[col].tolist(),
                x_pu=network.lines.x_pu[col],
                r_pu=network.lines.r_pu[col],
                g_pu=network.lines.g_pu[col],
                b_pu=network.lines.b_pu[col],
                s_nom_opt=network.lines.s_nom_opt[col],
                geom=network.lines.geom[col],
                topo=network.lines.topo[col]
            )
            session.add(res_line)
    except:
        for col in network.lines_t.p0:
            res_line = LineResult(
                result_id=new_res_id, 
                line_id=col,
                bus0=int(network.lines.bus0[col]),
                bus1=int(network.lines.bus1[col]),
                x=network.lines.x[col],
                r=network.lines.r[col],
                g=network.lines.g[col],
                b=network.lines.b[col],
                s_nom=network.lines.s_nom[col],
                s_nom_extendable=bool(network.lines.s_nom_extendable[col]),
                s_nom_min=network.lines.s_nom_min[col],
                s_nom_max=network.lines.s_nom_max[col],
                capital_cost=network.lines.capital_cost[col],
                length=network.lines.length[col],
                cables=int(network.lines.cables[col]),
                frequency=network.lines.frequency[col],
#                terrain_factor=network.lines.terrain_factor[col],
                p0=network.lines_t.p0[col].tolist(),
                p1=network.lines_t.p1[col].tolist(),
                x_pu=network.lines.x_pu[col],
                r_pu=network.lines.r_pu[col],
                g_pu=network.lines.g_pu[col],
                b_pu=network.lines.b_pu[col],
                s_nom_opt=network.lines.s_nom_opt[col],
                geom=network.lines.geom[col],
                topo=network.lines.topo[col]
            )
            session.add(res_line)
    session.commit()

    # insert active and reactive power of lines to database
    try:
        for col in network.transformers_t.p0:
            res_transformer = TransformerResult(
                result_id=new_res_id,
                trafo_id=col,
                bus0=int(network.transformers.bus0[col]),
                bus1=int(network.transformers.bus1[col]),
                x=network.transformers.x[col],
                r=network.transformers.r[col],
                g=network.transformers.g[col],
                b=network.transformers.b[col],
                s_nom=network.transformers.s_nom[col],
                s_nom_extendable=bool(network.transformers.s_nom_extendable[col]),
                s_nom_min=network.transformers.s_nom_min[col],
                s_nom_max=network.transformers.s_nom_max[col],
                tap_ratio=network.transformers.tap_ratio[col],
                phase_shift=network.transformers.phase_shift[col],
                capital_cost=network.transformers.capital_cost[col],
                p0=network.transformers_t.p0[col].tolist(),
                q0=network.transformers_t.q0[col].tolist(),
                p1=network.transformers_t.p1[col].tolist(),
                q1=network.transformers_t.q1[col].tolist(),
                x_pu=network.transformers.x_pu[col],
                r_pu=network.transformers.r_pu[col],
                g_pu=network.transformers.g_pu[col],
                b_pu=network.transformers.b_pu[col],
                s_nom_opt=network.transformers.s_nom_opt[col],
                geom=network.transformers.geom[col],
                topo=network.transformers.topo[col]
            )
            session.add(res_transformer)
    except:
        for col in network.transformers_t.p0:
            res_transformer = TransformerResult(
                result_id=new_res_id,
                trafo_id=col,
                bus0=int(network.transformers.bus0[col]),
                bus1=int(network.transformers.bus1[col]),
                x=network.transformers.x[col],
                r=network.transformers.r[col],
                g=network.transformers.g[col],
                b=network.transformers.b[col],
                s_nom=network.transformers.s_nom[col],
                s_nom_extendable=bool(network.transformers.s_nom_extendable[col]),
                s_nom_min=network.transformers.s_nom_min[col],
                s_nom_max=network.transformers.s_nom_max[col],
                tap_ratio=network.transformers.tap_ratio[col],
                phase_shift=network.transformers.phase_shift[col],
                capital_cost=network.transformers.capital_cost[col],
                p0=network.transformers_t.p0[col].tolist(),
                p1=network.transformers_t.p1[col].tolist(),
                x_pu=network.transformers.x_pu[col],
                r_pu=network.transformers.r_pu[col],
                g_pu=network.transformers.g_pu[col],
                b_pu=network.transformers.b_pu[col],
                s_nom_opt=network.transformers.s_nom_opt[col],
                geom=network.transformers.geom[col],
                topo=network.transformers.topo[col]
            )
            session.add(res_transformer)
    session.commit()
    
    # storage_units results
    try:
        for col in network.storage_units.p:
            res_sto = StorageResult(
                result_id=new_res_id,
                storage_id=col,
                bus=int(network.storage_units.bus[col]),
                dispatch=network.storage_units.dispatch[col],
                control=network.storage_units.control[col],
                p_nom=network.storage_units.p_nom[col],
                p_nom_extendable=bool(network.storage_units.p_nom_extendable[col]),
                p_nom_min=network.storage_units.p_nom_min[col],
                p_nom_max=network.storage_units.p_nom_max[col],
                p_min_pu_fixed=network.storage_units.p_min_pu[col],
                p_max_pu_fixed=network.storage_units.p_max_pu[col],
                sign=network.storage_units.sign[col],
#                source=network.storage_units.carrier[col],
                marginal_cost=network.storage_units.marginal_cost[col],
                capital_cost=network.storage_units.capital_cost[col],
                efficiency=network.storage_units.efficiency[col],
                soc_initial=network.storage_units.state_of_charge_initial[col],
                soc_cyclic=bool(network.storage_units.cyclic_state_of_charge[col]),
                max_hours=network.storage_units.max_hours[col],
                efficiency_store=network.storage_units.efficiency_store[col],
                efficiency_dispatch=network.storage_units.efficiency_dispatch[col],
                standing_loss=network.storage_units.standing_loss[col],
                p=network.storage_units_t.p[col].tolist(),
                q=network.storage_units_t.q[col].tolist(),
                state_of_charge=network.storage_units_t.state_of_charge[col].tolist(),
                spill=network.storage_units_t.spill[col].tolist(),
                p_nom_opt=network.storage_units.p_nom_opt[col]
            )
            session.add(res_sto)
    except:
        for col in network.storage_units_t.p:
            res_sto = StorageResult(
                result_id=new_res_id,
                storage_id=col,
                bus=int(network.storage_units.bus[col]),
                dispatch=network.storage_units.dispatch[col],
                control=network.storage_units.control[col],
                p_nom=network.storage_units.p_nom[col],
                p_nom_extendable=bool(network.storage_units.p_nom_extendable[col]),
                p_nom_min=network.storage_units.p_nom_min[col],
                p_nom_max=network.storage_units.p_nom_max[col],
                p_min_pu_fixed=network.storage_units.p_min_pu[col],
                p_max_pu_fixed=network.storage_units.p_max_pu[col],
                sign=network.storage_units.sign[col],
#                source=network.storage_units.carrier[col],
                marginal_cost=network.storage_units.marginal_cost[col],
                capital_cost=network.storage_units.capital_cost[col],
                efficiency=network.storage_units.efficiency[col],
                soc_initial=network.storage_units.state_of_charge_initial[col],
                soc_cyclic=bool(network.storage_units.cyclic_state_of_charge[col]),
                max_hours=network.storage_units.max_hours[col],
                efficiency_store=network.storage_units.efficiency_store[col],
                efficiency_dispatch=network.storage_units.efficiency_dispatch[col],
                standing_loss=network.storage_units.standing_loss[col],
                p=network.storage_units_t.p[col].tolist(),
                state_of_charge=network.storage_units_t.state_of_charge[col].tolist(),
                spill=network.storage_units_t.spill[col].tolist(),
                p_nom_opt=network.storage_units.p_nom_opt[col]
            )
            session.add(res_sto)
    session.commit()
    
    
if __name__ == '__main__':
    if pypsa.__version__ not in ['0.6.2', '0.8.0']:
        print('Pypsa version %s not supported.' % pypsa.__version__)
    pass
