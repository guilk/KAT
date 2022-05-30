import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r") as f:
    install_requires = list(f.read().splitlines())
 

setuptools.setup(
    name="KAT",
    version="0.1.0",
    description="KAT: A Knowledge Augmented Transformer for Vision-and-Language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=install_requires
)
