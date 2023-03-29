import re
from setuptools import setup


VERSION = '0.0.1'


if __name__ == '__main__':
    with open("README.md", 'r') as readme:
        description = readme.read()

    setup(
        name='bigraph-viz',
        version=VERSION,
        packages=[
            'bigraph_viz',
        ],
        author='Eran Agmon',
        author_email='agmon.eran@gmail.com',
        url='https://github.com/vivarium-collective/bigraph-viz',
        project_urls={},
        license='MIT',
        entry_points={
            'console_scripts': []},
        description=(),
        # long_description=long_description,
        # long_description_content_type='text/markdown',
        package_data={},
        include_package_data=True,
        python_requires='>=3.8, <3.11',
        install_requires=[
            'matplotlib>=3.5.1',
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering',
        ],
        keywords='bigraph multi-scale network visualization',
    )
