# -*- coding: utf-8 -*-


import pypsa
import pandas as pd

import sqlalchemy
from pypsa import io

from egoio.db_tables.calc_ego_hv_powerflow import Bus, Line, Generator, Load, Storage, Source, Transformer, TempResolution
from egoio.db_tables.calc_ego_hv_powerflow import BusVMagSet, GeneratorPqSet, LoadPqSet, StoragePqSet



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
# Time series generation
# -------------------------------------------------------
temp_id_set = 1
query = session.query(TempResolution.start_time).filter(TempResolution.temp_id == temp_id_set)
start_time = query.all()
start_time = ''.join(str(i) for i in start_time[0])

query = session.query(TempResolution.timesteps).filter(TempResolution.temp_id == temp_id_set)
periods = query.all()
periods =int(''.join(str(i) for i in  periods[0]))

query = session.query(TempResolution.resolution).filter(TempResolution.temp_id == temp_id_set)
frequency = query.all()
frequency = ''.join(str(i) for i in frequency[0])

time_range = pd.DatetimeIndex(freq=frequency,periods=periods,start=start_time)

# -------------------------------------------------------
# Time series data import
# -------------------------------------------------------
time_series_data = {BusVMagSet, GeneratorPqSet, LoadPqSet, StoragePqSet}

query = session.query(GeneratorPqSet.generator_id, GeneratorPqSet.p_set)
gen_p_set = pd.read_sql_query(query.statement,session.bind,index_col="generator_id")

query = session.query(BusVMagSet)
busvmagset_data = pd.read_sql_query(query.statement,session.bind,index_col="bus_id")

# -------------------------------------------------------
# Time series data manipulation for PyPSA
# -------------------------------------------------------

gen_p_set.index = [str(i) for i in gen_p_set.index]
gen_p_set = gen_p_set['p_set'].apply(pd.Series).transpose().set_index(time_range)


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

io.import_series_from_dataframe(network,gen_p_set,"Generator","p_set")

now = network.snapshots[5:7]
print(network.generators_t.p_set)

network.pf(now)

session.close()