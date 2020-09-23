from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
PACKAGES=['numpy',
            'opencv-python',
            'scikit-image',
            'scipy',
            'pandas',
            'scikit-learn',
            'SimpleITK',
            'matplotlib',
            'torch',
            'airlab',
            'fire',
            'nibabel',
            'pathflowai']

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

setup(name='pathflow_mixmatch',
      version='0.1',
      description='Don\'t mix, match! Simple utilities for improved registration of Histopathology Whole Slide Images.',
      url='https://github.com/jlevy44/PathFlow-MixMatch',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=['bin/install_mixmatch_dependencies'],
      entry_points={
            'console_scripts':['pathflow-mixmatch=pathflow_mixmatch.cli:main']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['pathflow_mixmatch'],
      install_requires=PACKAGES)
