"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.
Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line."""

__copyright__ = "Reiner Lemoine Institut, ZNES, Next Energy, IKS Uni Magdeburg"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "gplssm"


from setuptools import find_packages, setup

setup(name='egopowerflow',
      author='openego development group',
      author_email='oemof@rl-institut.de',
      description='Powerflow analysis based on PyPSA',
      version='0.0.5',
      license="GNU Affero General Public License Version 3 (AGPL-3.0)",
      packages=find_packages(),
      include_package_data=True,
      install_requires=['pandas >= 0.17.0, <=0.19.1',
                        'pypsa >= 0.8.0, <= 0.8.0',
                        'sqlalchemy >= 1.0.15, <= 1.1.4',
                        'oemof.db >=0.0.4, <=0.0.4',
                        'geoalchemy2 >= 0.3.0, <=0.4.0',
                        'matplotlib >= 1.5.3, <=1.5.3']
     )
