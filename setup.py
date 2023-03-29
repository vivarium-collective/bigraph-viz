from setuptools import setup, find_packages


VERSION = '0.0.1'


if __name__ == '__main__':
    with open("README.md", 'r') as readme:
        description = readme.read()

    setup(
        name='bigraph-viz',
        version=VERSION,
        packages=find_packages,
        author='Eran Agmon',
        author_email='agmon.eran@gmail.com',
        url='https://github.com/vivarium-collective/bigraph-viz',
        project_urls={},
        license='MIT',
        entry_points={
            'console_scripts': []},
        description='plotting tool for compositional bigraph schema',
        long_description=description,
        long_description_content_type='text/markdown',
        keywords='bigraph multi-scale network visualization',
        package_data={},
        include_package_data=True,
        classifiers=[
            'Development Status :: 3 - Alpha',
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
        python_requires='>=3.8',
        install_requires=[
            'graphviz',
        ],
    )
