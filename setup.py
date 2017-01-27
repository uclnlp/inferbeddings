# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
      requirements = f.readlines()

setup(name='inferbeddings',
      version='0.1.0',
      description='Rule Injection in Knowledge Graph Embeddings via Adversarial Training',
      author='Pasquale Minervini',
      author_email='p.minervini@cs.ucl.ac.uk',
      url='https://github.com/uclmr/inferbeddings',
      test_suite='tests',
      license='MIT',
      install_requires=requirements,
      setup_requires=['pytest-runner'] + requirements,
      tests_require=requirements,
      packages=find_packages())
