from setuptools import setup, find_packages

setup(
  name='cy2path',
  version='0.0.1',
  description='State probability simulations based trajectory inference.',
  license='BSD 3-Clause License',
  packages=find_packages(),
  install_requires=[
      'scvelo',
      'seaborn',
      'scikit-learn>=1.3.0',
      'imageio==2.19.3',
      'tqdm',
      'hausdorff',
      'dtaidistance',
      'torch',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ])
