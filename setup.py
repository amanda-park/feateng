from setuptools import setup

setup(name='feateng',
      version='0.1',
      description='A library to clean and preprocess your data in Python.',
      url='https://bitbucket.spectrum-health.org:7991/stash/projects/QSE/repos/python-feature-engineering/',
      author='Amanda Park & Phil Sattler',
      author_email='amanda.park@spectrumhealth.org',
      license='MIT',
      packages=['feateng'],
      install_requires=[
            'pandas',
            'numpy',
            'scipy',
            'ppscore'
        ],
      zip_safe=False)