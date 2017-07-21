# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
      setup_requires = f.readlines()

setup(name='inferbeddings',
      version='0.4.0',
      description='Injecting Background Knowledge in Neural Models via Adversarial Set Regularisation',
      author='Pasquale Minervini',
      author_email='p.minervini@ucl.ac.uk',
      url='https://github.com/uclmr/inferbeddings',
      test_suite='tests',
      license='MIT',
      install_requires=setup_requires,
      setup_requires=setup_requires,
      tests_require=setup_requires,
      packages=find_packages())
