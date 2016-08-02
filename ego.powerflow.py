# -*- coding: utf-8 -*-


import pypsa
import pandas as pd

import sqlalchemy
from pypsa import io

from calc_ego_hv_powerflow import Bus, Line, Generator, Load, Storage, Source, Transformer, TempResolution
from calc_ego_hv_powerflow import BusVMagSet, GeneratorPqSet, LoadPqSet, StoragePqSet



# -------------------------------------------------------
# Connection to DB
# -------------------------------------------------------

print('Please provide connection parameters to database:')

host = input('host (default 127.0.0.1): ') or '127.0.0.1'
port = input('port (default 5432): ') or '5432'
user = input('user (default postgres): ') or 'postgres'
database = input('database name: ')
password = input('password: ') 

con = sqlalchemy.create_engine('postgresql://' + 
                                '%s:%s@%s:%s/%s' % (user, password, host, port, database))


Session = sqlalchemy.orm.sessionmaker(bind=con)
session = Session()

# -------------------------------------------------------
# Components import
# -------------------------------------------------------
tablenames = [Bus, Line, Generator, Load, Storage, Source]

component_data = {}

for tablename in tablenames:
    component_data[tablename] = pd.read_sql_query(session.query(tablename).statement,session.bind, index_col = str(tablename.__name__).lower()+ "_id")

component_data[Transformer] = pd.read_sql_query(session.query(Transformer).statement,session.bind,index_col="trafo_id")


# -------------------------------------------------------
# Time series data import
# -------------------------------------------------------
time_series_data = {BusVMagSet, GeneratorPqSet, LoadPqSet, StoragePqSet}

query = session.query(GeneratorPqSet.generator_id, GeneratorPqSet.q_set)
gen_q_set = pd.read_sql_query(query.statement,session.bind,index_col="generator_id")

query = session.query(BusVMagSet)
busvmagset_data = pd.read_sql_query(query.statement,session.bind,index_col="bus_id")

# -------------------------------------------------------
# Time series generation
# -------------------------------------------------------
query = session.query(TempResolution.start_time).filter(TempResolution.temp_id == 1)
start_time = query.all()
start_time = ''.join(str(i) for i in start_time).replace(",","").replace("(","").replace(")","")

time_range = pd.DatetimeIndex(freq="h",periods=10,start=start_time)
gen_q_set.index = [str(i) for i in gen_q_set.index]
gen_q_set2 = gen_q_set['q_set'].apply(pd.Series).transpose().set_index(time_range)


# -------------------------------------------------------
# PyPSA network init
# -------------------------------------------------------

network = pypsa.Network()
network.set_snapshots(time_range)


network.import_components_from_dataframe(component_data[Bus],'Bus')
network.import_components_from_dataframe(component_data[Line],'Line')
network.import_components_from_dataframe(component_data[Transformer],'Transformer')
network.import_components_from_dataframe(component_data[Generator],'Generator')
network.import_components_from_dataframe(component_data[Load],'Load')

io.import_series_from_dataframe(network,gen_q_set2,"Generator","q_set")

now = network.snapshots[5:7]
print(network.generators_t.q_set)

network.pf(now)

session.close()