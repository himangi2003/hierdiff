"""
This setup.py allows your python package to be installed. 
Please completely update the parameters in the opts dictionary. 
For more information, see https://stackoverflow.com/questions/1471994/what-is-setup-py
"""

from setuptools import setup, find_packages
PACKAGES = find_packages()

"""Read the contents of your README file"""
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

opts = dict(name='hierdiff',
            maintainer='Andrew Fiore-Gartland',
            maintainer_email='agartlan@fredhutch.org',
            description='Clustering high-dimensional instances (e.g. T cell receptors) and testing whether clusters of instances are differentially abundant in two or more categorical conditions, with interactive tree visualization.',
            long_description=long_description,
            long_description_content_type='text/markdown',
            url='https://github.com/agartland/hierdiff',
            license='MIT',
            author='Andrew Fiore-Gartland',
            author_email='agartlan@fredhutch.org',
            version='0.9',
            packages=PACKAGES,
            include_package_data=True
           )

install_reqs = [  'numpy>=1.18.1',
                  'pandas>=0.24.2',
                  'scipy>=1.4.1',
                  'jinja2>=2.10.1',
                  'statsmodels>=0.10',
                  'fishersapi>=0.1.2',
                  'joblib>=0.16']

if __name__ == "__main__":
    setup(**opts, install_requires=install_reqs)

