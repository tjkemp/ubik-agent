import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ubik-agent",
    version="0.0.1",
    author="tjkemp",
    author_email="tero.kemppi@gmail.com",
    description="A small Reinforcement Learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tjkemp/ubik-agent",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 2 - Pre-Alpha"
    ],
    python_requires='>=3.6',
)
