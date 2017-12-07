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
      extras_require={
            'tensorflow': ['tensorflow>=1.4.0'],
            'tensorflow_gpu': ['tensorflow-gpu>=1.4.0'],
      },
      setup_requires=setup_requires,
      tests_require=setup_requires,
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules'
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages(),
      keywords='tensorflow machine learning adversarial training knowledge graphs')
