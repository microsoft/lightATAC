from distutils.core import setup

setup(
    name='lightATAC',
    version='0.1.0',
    author='Ching-An Cheng',
    author_email='chinganc@microsoft.com',
    packages=['lightATAC'],
    url='https://github.com/chinganc/lightATAC',
    license='MIT LICENSE',
    description='A lightweight version of ATAC code',
    long_description=open('README.md').read(),
    install_requires=[
        "gym==0.17.2",
        "torch==1.12.1",
        "tensorboard==2.10.0",
        "psutil==5.9.1",
        "protobuf==3.19.4",
        "dowel==0.0.4",
        "d4rl@git+https://github.com/rail-berkeley/d4rl@6330b4e09e36a80f4b706a3885d59d97745c05a9"
    ]
)