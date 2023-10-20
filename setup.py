from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='yid_langchain_extensions',
    version='0.4.30',
    author='Dmitrii Rashchenko',
    author_email='dimitree54@gmail.com',
    packages=find_packages(),
    description='Useful classes extending langchain library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dimitree54/yid_langchain_extensions',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=required
)
