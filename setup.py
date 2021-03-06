import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyfizi",
    version="0.7",
    author="Nicholas Mancuso, Megan Roytman",
    author_email="nick.mancuso@gmail.com, meganroytman@gmail.com",
    description="Impute GWAS summary statistics using reference genotype data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bogdanlab/fizi",
    packages=["pyfizi"],
    python_requires='>=3',
    install_requires=[
        "numpy>=1.14.5",
        "scipy",
        "pandas>=0.23.3",
        "pandas-plink",
      ],
    scripts=[
        "bin/fizi",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
)
