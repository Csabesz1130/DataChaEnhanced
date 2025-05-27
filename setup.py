from setuptools import setup, find_packages

setup(
    name="DataChaEnhanced",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'requests',
        'packaging',
        'numpy==1.24.3',
        'scipy==1.10.1',
        'matplotlib',
        'pandas',
        'PyWavelets==1.4.1',
        'openpyxl'
    ]
)