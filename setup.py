#!/usr/bin/env python

from setuptools import find_packages, setup

install_requires = [
    # we're hiding in the conda environment for now ...
]

__version__ = "0.1.0"
# project_dir = os.path.abspath(__file__)
# version_file = os.path.join(project_dir, trams', 'version.py')
# with open(version_file) as f:
#    exec(f.read())


# Some duplication of properties here and in package.xml.
# Make sure to update them both.
# That is the price paid for a pypi and catkin package.
subpackages = ["tristan", "triceps", "radutils"]
d = setup(
    name="guardian",
    version=__version__,
    packages=find_packages(exclude=["tests*", "docs*", *(f"{package}*" for package in subpackages)]),
    install_requires=install_requires,
    author="Guy Rosman, Stephen McGill, Simon Stent, Luke Fletcher",
    maintainer="Guy Rosman <guy.rosman@tri.global>",  # any volunteers?
    url="https://github.com/ToyotaResearchInstitute/risk_aware_driving/",
    keywords="guardian",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Testing :: Traffic Generation",  # irony ...
    ],
    description="pythonic playground",
    long_description="Joint repo for general guardian research code.",
    license="BSD",
    test_suite="nose.collector",
    tests_require=["nose"],
    # add..
    scripts=[],
)
