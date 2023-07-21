from setuptools import setup, find_packages

setup(
  name='cy2path',
  version='0.0.1',
  license='GNU General Public License v3 (GPLv3)',
  description='Lineage inference with factorial latent dynamic models trained on Markovian simulations of biological processes using single cell RNA sequencing data.',
  author = 'Revant Gupta',                   
  author_email = 'revant.gupta.93@gmail.com',
  packages=find_packages(),
  keywords = ['Lineage inference', 'single-cell RNA sequencing', 
              'state space models', 'markov chain simulation'],
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
