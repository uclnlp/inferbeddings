# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
      setup_requires = f.readlines()

setup(name='inferbeddings',
      version='0.3.0',
      description='Adversarial Set Regularizaton for '
                  'Knowledge Base Population and '
                  'Natural Language Inference',
      author='Pasquale Minervini',
      author_email='p.minervini@ucl.ac.uk',
      url='https://github.com/uclmr/inferbeddings',
      test_suite='tests',
      license='MIT',
      install_requires=setup_requires,
      setup_requires=setup_requires,
      tests_require=setup_requires,
      packages=find_packages())
