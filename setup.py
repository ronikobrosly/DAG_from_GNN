import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DAG_from_GNN",
    version="0.0.1",
    author="Roni Kobrosly",
    author_email="roni.kobrosly@gmail.com",
    description='Forked from https://github.com/fishmoon1234/DAG-GNN, this is a clean implementation of Yu and Chen et al.s "DAG Structure Learning with Graph Neural Networks" method',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ronikobrosly/DAG_from_GNN",
    packages=setuptools.find_packages(include=['DAG_from_GNN']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.6',
    install_requires=[
        'matplotlib',
        'netgraph',
        'networkx',
        'numpy',
        'pandas',
        'scipy',
        'torch'
    ]
)
