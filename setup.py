import os
from glob import glob
from setuptools import setup

package_name = 'custom_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[
        'custom_controller.impedance_controller',
    ],
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.urdf'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Courtney McBeth',
    maintainer_email='cmcbeth2@illinois.edu',
    description='ROS2 impedance controller for Jazzy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'impedance_controller = custom_controller.impedance_controller:main'
        ],
    },
)
