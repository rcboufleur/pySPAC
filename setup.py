import setuptools

# Read the contents of your README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyspac",
    version="0.1.0",
    author="Rodrigo C. Boufleur",
    author_email="rcboufleur@gmail.com",
    description="A library for fitting and analyzing asteroid phase curves.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Specify the license using the modern SPDX identifier
    license="MIT",
    url="https://github.com/rcboufleur/pySPAC",
    project_urls={
        "Bug Tracker": "https://github.com/rcboufleur/pySPAC/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    py_modules=["pyspac", "constants"],
    python_requires=">=3.6",
    # List the package dependencies directly
    install_requires=[
        "numpy",
        "lmfit",
        "sbpy",
    ],
)
