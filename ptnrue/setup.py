import setuptools

setuptools.setup(
    name="ptnrue",
    version="0.0.1",
    author="Riccardo Fiorista",
    author_email="rfiorista@uva.nl",
    description="A package for Public Transport Network Reduction Under Equality",
    long_description_content_type="text/markdown",
    url="https://github.com/NONE_YET",
    project_urls={
        "Bug Tracker": "https://github.com/NONE_YET/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)