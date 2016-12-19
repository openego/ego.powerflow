"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.
Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line."""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "gplssm"


from setuptools import find_packages, setup

setup(name='egopowerflow',
      author='openego development group',
      description='Powerflow analysis based on PyPSA',
      packages=find_packages(),
      install_requires=['pandas >= 0.17.0',
                        'pypsa <= 0.6.2',
                        'sqlalchemy',
                        # 'egoio', # comment til release of ego.io
                        # 'oemof.db',
                        'geoalchemy2',
                        'matplotlib'] #to be installed manually
     )