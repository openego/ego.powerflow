from setuptools import find_packages, setup

setup(name='egopowerflow',
      author='openego development group',
      description='Powerflow analysis based on PyPSA',
      packages=find_packages(),
      install_requires=['pandas >= 0.17.0',
                        'pypsa',
                        'sqlalchemy',
                        # 'egoio', # comment til release of ego.io
                        # 'oemof.db',
                        'geoalchemy2',
                        'matplotlib'] #to be installed manually
     )