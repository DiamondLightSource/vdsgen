from setuptools import setup, find_packages

MODULE_NAME = "vdsgen"

setup(
    name=MODULE_NAME,
    version='0.2.3',
    description='Creates virtual dataset HDF5 files',
    long_description=open("README.rst").read(),
    url='https://github.com/dls-controls/vds-gen',
    author='Gary Yendell',
    author_email='gary.yendell@diamond.ac.uk',
    keywords='',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
    ],
    license='APACHE',
    include_package_data=True,
    test_suite='nose.collector',
    install_requires=[
        'h5py>=2.7.1',
        'numpy>=1.11.1',
        'Cython>=0.19.1',
        'six'
    ],
    tests_require=[
        'nose>=1.3.0',
        'mock'
    ],
    zip_safe=False,
    entry_points={'console_scripts': ["dls-vds-gen.py = vdsgen.app:main"]}
)
