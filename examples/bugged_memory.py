# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:02:56 2016
"""

from tools.pypsa_io import oedb_session,\
    get_timerange, import_components, create_powerflow_problem
import pypsa
import pandas as pd

from egoio.db_tables.calc_ego_hv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution
    


session = oedb_session()

scenario = 'Status Quo'

switch = 1 # 1 for "old" version with normal memory usage. 
           # Anything else for bugged import that causes high mem usage

if switch == 1:
    tablenames = [Bus, Line, Generator, Load]
    
    component_data = {}
    
    for tablename in tablenames:
        component_data[tablename] = pd.read_sql_query(session.query(tablename).statement,session.bind, index_col = str(tablename.__name__).lower()+ "_id")
    
    component_data[Transformer] = pd.read_sql_query(session.query(Transformer).statement,session.bind,index_col="trafo_id")
    
    network = pypsa.Network()
    
    network.import_components_from_dataframe(component_data[Bus],'Bus')
    network.import_components_from_dataframe(component_data[Line],'Line')
    network.import_components_from_dataframe(component_data[Transformer],'Transformer')
    network.import_components_from_dataframe(component_data[Generator],'Generator')
    network.import_components_from_dataframe(component_data[Load],'Load')
    
else:
    
    temp_id_set = 1

    timerange = get_timerange(session, temp_id_set, TempResolution)

    # define relevant tables
    tables = [Bus, Line, Generator, Load, Transformer]
    
    # get components from database tables
    components = import_components(tables, session, scenario)
    
    # create PyPSA powerflow problem
    network, snapshots = create_powerflow_problem(timerange, components)