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
# Settings
# -------------------------------------------------------

start_h = 1 # Start and end hour (of year) used for calculations (min = 0, max = 8760)
end_h = 2

gen_pq_set_sw = False
bus_v_mag_set_sw = False
load_pq_set_sw = False
storage_pq_set_sw = False

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
tr_start = query.all()
tr_start = ''.join(str(i) for i in tr_start[0])

query = session.query(TempResolution.timesteps).filter(TempResolution.temp_id == temp_id_set)
periods = query.all()
periods =int(''.join(str(i) for i in  periods[0]))

query = session.query(TempResolution.resolution).filter(TempResolution.temp_id == temp_id_set)
frequency = query.all()
frequency = ''.join(str(i) for i in frequency[0])

time_range_total = pd.DatetimeIndex(freq=frequency,periods=periods,start=tr_start)
time_range_lim = time_range_total[start_h-1:end_h]

# -------------------------------------------------------
# Time series data import
# -------------------------------------------------------

if gen_pq_set_sw == True:
    query = session.query(GeneratorPqSet.generator_id, 
                          GeneratorPqSet.p_set[start_h:end_h],
                          GeneratorPqSet.q_set[start_h:end_h],
                          GeneratorPqSet.p_min_pu[start_h:end_h],
                          GeneratorPqSet.p_max_pu[start_h:end_h])
    gen_pq_set = pd.read_sql_query(query.statement,session.bind,index_col="generator_id")
    gen_pq_set.columns = ["p_set","q_set","p_min_pu","p_max_pu"]

if bus_v_mag_set_sw == True:
    query = session.query(BusVMagSet.bus_id, 
                          BusVMagSet.v_mag_pu_set[start_h:end_h])
    bus_v_mag_set = pd.read_sql_query(query.statement,session.bind,index_col="bus_id")
    bus_v_mag_set.columns = ["v_mag_pu_set"]

if load_pq_set_sw == True:
    query = session.query(LoadPqSet.load_id, 
                          LoadPqSet.p_set[start_h:end_h], 
                          LoadPqSet.q_set[start_h:end_h])
    load_pq_set = pd.read_sql_query(query.statement,session.bind,index_col="load_id")
    load_pq_set.columns = ["p_set","q_set"]

if storage_pq_set_sw == True:
    query = session.query(StoragePqSet.storage_id, 
                          StoragePqSet.p_set[start_h:end_h],
                          StoragePqSet.q_set[start_h:end_h],
                          StoragePqSet.p_min_pu[start_h:end_h],
                          StoragePqSet.p_max_pu[start_h:end_h], 
                          StoragePqSet.soa_set[start_h:end_h],
                          StoragePqSet.inflow[start_h:end_h])
    storage_pq_set = pd.read_sql_query(query.statement,session.bind,index_col="storage_id")
    storage_pq_set.columns = ["p_set","q_set","p_min_pu","p_max_pu","soa_set","inflow"]

# -------------------------------------------------------
# Time series data manipulation for PyPSA
# -------------------------------------------------------
# -------------------------------------------------------
# gen_pq_set
# -------------------------------------------------------

if gen_pq_set_sw == True:
    gen_p_set = gen_pq_set["p_set"]
    gen_p_set.index = [str(i) for i in gen_p_set.index]
    gen_p_set = gen_p_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    gen_q_set = gen_pq_set["q_set"]
    gen_q_set.index = [str(i) for i in gen_q_set.index]
    gen_q_set = gen_q_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    gen_p_min_pu_set = gen_pq_set["p_min_pu"]
    gen_p_min_pu_set.index = [str(i) for i in gen_p_min_pu_set.index]
    gen_p_min_pu_set = gen_p_min_pu_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    gen_p_max_pu_set = gen_pq_set["p_max_pu"]
    gen_p_max_pu_set.index = [str(i) for i in gen_p_max_pu_set.index]
    gen_p_max_pu_set = gen_p_max_pu_set.apply(pd.Series).transpose().set_index(time_range_lim)

# -------------------------------------------------------
# bus_v_mag_set
# -------------------------------------------------------

if bus_v_mag_set_sw == True:
    bus_v_mag_set.index = [str(i) for i in bus_v_mag_set.index]
    bus_v_mag_set = bus_v_mag_set.apply(pd.Series).transpose().set_index(time_range_lim)

# -------------------------------------------------------
# load_pq_set
# -------------------------------------------------------

if load_pq_set_sw == True:
    load_p_set = load_pq_set["p_set"]
    load_p_set.index = [str(i) for i in load_p_set.index]
    load_p_set = load_p_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    load_q_set = load_pq_set["q_set"]
    load_q_set.index = [str(i) for i in load_q_set.index]
    load_q_set = load_q_set.apply(pd.Series).transpose().set_index(time_range_lim)

# -------------------------------------------------------
# storage_pq_set
# -------------------------------------------------------

if storage_pq_set_sw == True:
    storage_p_set = storage_pq_set["p_set"]
    storage_p_set.index = [str(i) for i in storage_p_set.index]
    storage_p_set = storage_p_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    storage_q_set = storage_pq_set["q_set"]
    storage_q_set.index = [str(i) for i in storage_q_set.index]
    storage_q_set = storage_q_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    storage_p_min_pu_set = storage_pq_set["p_min_pu"]
    storage_p_min_pu_set.index = [str(i) for i in storage_p_min_pu_set.index]
    storage_p_min_pu_set = storage_p_min_pu_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    storage_p_max_pu_set = storage_pq_set["p_max_pu"]
    storage_p_max_pu_set.index = [str(i) for i in storage_p_max_pu_set.index]
    storage_p_max_pu_set = storage_p_max_pu_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    storage_soa_set = storage_pq_set["soa_set"]
    storage_soa_set.index = [str(i) for i in storage_soa_set.index]
    storage_soa_set = storage_soa_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
    storage_inflow_set = storage_pq_set["inflow"]
    storage_inflow_set.index = [str(i) for i in storage_inflow_set.index]
    storage_inflow_set = storage_inflow_set.apply(pd.Series).transpose().set_index(time_range_lim)
    
# -------------------------------------------------------
# PyPSA network init
# -------------------------------------------------------

network = pypsa.Network()
network.set_snapshots(time_range_lim)
now = network.snapshots

# -------------------------------------------------------
# add static components to PyPSA network 
# -------------------------------------------------------

network.import_components_from_dataframe(component_data[Bus],'Bus')
network.import_components_from_dataframe(component_data[Line],'Line')
network.import_components_from_dataframe(component_data[Transformer],'Transformer')
network.import_components_from_dataframe(component_data[Generator],'Generator')
network.import_components_from_dataframe(component_data[Load],'Load')
network.import_components_from_dataframe(component_data[Storage],'Storage')

# -------------------------------------------------------
# add time series of components to PyPSA network 
# -------------------------------------------------------

if gen_pq_set_sw == True:
    io.import_series_from_dataframe(network,gen_p_set,"Generator","p_set")
    io.import_series_from_dataframe(network,gen_q_set,"Generator","q_set")
    io.import_series_from_dataframe(network,gen_p_min_pu_set,"Generator","p_min_pu")
    io.import_series_from_dataframe(network,gen_p_max_pu_set,"Generator","p_max_pu")

if bus_v_mag_set_sw == True:
    io.import_series_from_dataframe(network,bus_v_mag_set,"Bus","v_mag_pu_set")

if load_pq_set_sw == True:
    io.import_series_from_dataframe(network,load_p_set,"Load","p_set")
    io.import_series_from_dataframe(network,load_q_set,"Load","q_set")
    
if storage_pq_set_sw == True:
    io.import_series_from_dataframe(network,storage_p_set,"Storage","p_set")
    io.import_series_from_dataframe(network,storage_q_set,"Storage","q_set")
    io.import_series_from_dataframe(network,storage_p_min_pu_set,"Storage","p_min_pu")
    io.import_series_from_dataframe(network,storage_p_max_pu_set,"Storage","p_max_pu")   
    io.import_series_from_dataframe(network,storage_soa_set,"Storage","state_of_charge_set")    
    io.import_series_from_dataframe(network,storage_inflow_set,"Storage","inflow")     

# -------------------------------------------------------
# AC-PF calculations
# -------------------------------------------------------  

network.pf(now)

# -------------------------------------------------------
# sqlalchemy session close
# ------------------------------------------------------- 

session.close()