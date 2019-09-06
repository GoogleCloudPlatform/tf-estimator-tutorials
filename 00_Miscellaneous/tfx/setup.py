from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
 'tensorflow==1.14.0',
 'tensorflow-data-validation==0.11.0'
]

setup(
    name='tfdv-data-extraction',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True
)