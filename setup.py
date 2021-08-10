import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyfizi",
    version="0.7.1",
    author="Nicholas Mancuso, Megan Roytman",
    author_email="nicholas.mancuso@med.usc.edu",
    description="Impute GWAS summary statistics using reference genotype data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bogdanlab/fizi",
    packages=["pyfizi"],
    python_requires='>=3',
    install_requires=[
        "numpy>=1.14.5",
        "scipy>=1.2.0",
        "pandas>=1.2.0",
        "pandas-plink",
      ],
    scripts=[
        "bin/fizi",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
