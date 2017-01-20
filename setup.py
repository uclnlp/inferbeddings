# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

setup(name='inferbeddings',
      version='0.1.0',
      description='Rule Injection in Knowledge Graph Embeddings via Adversarial Training',
      author='Pasquale Minervini',
      author_email='p.minervini@cs.ucl.ac.uk',
      url='https://github.com/uclmr/inferbeddings',
      test_suite='tests',
      license='MIT',
      install_requires=[
            'tensorflow>=0.8',
            'parsimonious>=0.7',
            'terminaltables>=2.0',
            'colorclass>=2.0'
      ],
      packages=find_packages())
