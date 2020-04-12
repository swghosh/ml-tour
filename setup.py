import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-tour",
    version="0.0.1",
    author="Swarup Ghosh",
    author_email="codecrafts.cf@icloud.com",
    description="A small library of curated machine learning algorithms implemented using TensorFlow 2.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swghosh/ml-tour",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Information Technology",
        "Topic :: Utilities"
    ],
    python_requires='>=3.5',
)