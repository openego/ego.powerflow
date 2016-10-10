from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


def oedb_session(section='oedb'):
    """Get SQLAlchemy session object with valid connection to OEDB"""

    # get session object by oemof.db tools (requires .oemof/config.ini
    try:
        from oemof import db
        conn = db.connection(section=section)

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