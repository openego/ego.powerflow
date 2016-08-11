import pypsa
import pandas as pd

from sqlalchemy.orm import sessionmaker, load_only
from sqlalchemy import create_engine
from pypsa import io
from oemof import db

from egoio.db_tables.calc_ego_mv_powerflow import Bus, Line, Generator, Load, \
    Transformer, TempResolution, BusVMagSet, GeneratorPqSet, LoadPqSet

def oedb_session():
    """Get SQLAlchemy session object with valid connection to OEDB"""

    # get session object by oemof.db tools (requires .oemof/config.ini
    try:
        conn = db.connection(section='oedb')

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



if __name__ == '__main__':
    session = oedb_session()

    gen_cols = ['temp_id', 'p_set', 'q_set']

    gen_pq_set = get_pq_sets(session, GeneratorPqSet, index_col='generator_id',
                             columns=gen_cols)
    print(gen_pq_set)
