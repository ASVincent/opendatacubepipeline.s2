#!/usr/bin/env python

from setuptools import setup, find_packages

import versioneer

setup(name='tesp',
      version=versioneer.get_version(),
      description=('A temporary solution to get packaging underway. '
                   'Code will eventually be ported eo-datasets.'),
      url='https://github.com/OpenDataCubePipelines/opendatacubepipeline.tesp',
      packages=find_packages(),
      install_requires=[
          'click',
          'click_datetime',
          'folium',
          'geopandas',
          'h5py',
          'luigi',
          'numpy',
          'pathlib',
          'pyyaml',
          'rasterio',
          'scikit-image',
          'shapely',
          'structlog',
          'eodatasets',
          'checksumdir',
          'eugl',
      ],
      dependency_links=[
          'git+https://github.com/GeoscienceAustralia/eo-datasets@develop#egg=eodatasets-0.1dev',
          'git+https://github.com/OpenDataCubePipelines/eugl.git#egg=eugl-0.0.2'
      ],
      scripts=['bin/s2package', 'bin/ard_pbs', 'bin/search_s2'],
      include_package_data=True)
