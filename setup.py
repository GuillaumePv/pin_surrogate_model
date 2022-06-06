import setuptools
from distutils.core import setup

setup(
  name = 'DeepSurrogate_pin',
  packages = ['model'],
  version = '0.2',
  license='MIT',
  setup_requires=['wheel'],
  description = 'Deep surrogate model for the probability of informed trading model',
  author = 'Guillaume Pavé',
  author_email = 'guillaumepave@gmail.com',
  url = 'https://github.com/GuillaumePv/',
  # download_url = 'https://github.com/AntoineDidisheim/didipack/archive/v0.1.1.tar.gz',    # I explain this later on
  keywords = ['Machine learning'],
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'tensorflow'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)