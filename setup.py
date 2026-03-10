from setuptools import setup


setup(
    name="cosmocnc_jax",
    version="1.0",
    description="JAX-accelerated Python package for fast cluster number count likelihood computation",
    zip_safe=False,
    packages=["cosmocnc_jax"],
    author = 'Inigo Zubeldia and Boris Bolliet',
    author_email = 'inigo.zubeldia@ast.cam.ac.uk',
    url = 'https://github.com/inigozubeldia/cosmocnc',
    download_url = 'https://github.com/inigozubeldia/cosmocnc',
    package_data={
    },
)
