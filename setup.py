from setuptools import setup

setup(name='feateng',
      version='0.1',
      description='A library to clean and preprocess your data in Python.',
      url='https://github.com/amanda-park/feateng',
      author='Amanda Park & Phil Sattler',
      author_email='apark24@binghamton.edu',
      license='MIT',
      packages=['feateng'],
      install_requires=[
            'pandas',
            'numpy',
            'scipy',
            'ppscore'
        ],
      zip_safe=False)