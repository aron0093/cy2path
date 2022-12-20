from setuptools import setup, find_packages

setup(
  name='switchstate',
  version='0.0.1',
  description='State probability simulations based trajectory inference.',
  license='BSD 3-Clause License',
  packages=find_packages(),
  install_requires=[
      'scvelo',
      'imageio==2.19.3',
      'seaborn',
      'hausdorff',
      'networkit',
      'torch',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ])