import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyfizi",
    version="0.5",
    author="Nicholas Mancuso, Megan Roytman",
    author_email="nick.mancuso@gmail.com, meganroytman@gmail.com",
    description="Impute GWAS summary statistics using reference genotype data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bogdanlab/fizi",
    packages=["pyfizi"],
    install_requires=[
          "numpy",
          "scipy",
          "pandas>=0.17.0",
          "pandas-plink",
      ],
    scripts=[
        "bin/fizi.py",
    ],
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
)
