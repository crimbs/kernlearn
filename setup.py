from setuptools import setup

setup(
    name="kernlearn",
    version="0.1",
    description="Learn interaction kernels that govern collective behaviour",
    url="http://github.com/crimbs/kernlearn",
    author="Christian Hines",
    author_email="christian.hines20@imperial.ac.uk",
    license="MIT",
    packages=["kernlearn"],
    install_requires=[
        "jax",
        "jaxlib",
        "optax",
        "dm-haiku",
        "matplotlib",
        "numpy",
        "tqdm",
    ],
    zip_safe=False,
)
