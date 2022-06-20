import setuptools
from distutils.core import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

setup(
  name = 'DeepSurrogatepin',
  packages = ['model'],
  version = '1.01',
  license='MIT',
  setup_requires=['wheel'],
  description = 'Deep surrogate model for the probability of informed trading model',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Guillaume Pavé',
  author_email = 'guillaumepave@gmail.com',
  url = 'https://github.com/GuillaumePv/pin_surrogate_model',
  # download_url = 'https://github.com/AntoineDidisheim/didipack/archive/v0.1.1.tar.gz',    # I explain this later on
  keywords = ['Machine learning, market microstructure'],
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'tensorflow',
          'tqdm',
          'tensorflow_probability'
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
  include_package_data=True,
    package_data={'': ['./model_save/Layer_400_400_200_100_swish_Lr0_0005_ADAMoMSE_BATCH_256tr_size_3CM/*.p']},
)