from setuptools import setup

setup(
    name='ASHRAE140Automation',
    version='0.0.0',
    packages=['src'],
    url='https://github.com/john-grando/ashrae-140-automation',
    license='',
    author='GARD Analytics and NREL for US DOE',
    author_email='',
    description='Automation of ASHRAE 140 Testing Verification',
    entry_points={
        'console_scripts': ['test_input_processing=src.test_input_processing:main'],
    }
)
