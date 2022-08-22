from distutils.core import setup
from typing import List
from setuptools import find_packages


def read_version(filename: str) -> str:
    """Read the version number from a file.
    Args:
        filename (str): The file to read the version number from.
    Returns:
        str: The version number.
    """
    with open(filename, "r") as v:
        major_minor_patch = v.read().strip()
    return major_minor_patch


def get_requirements(filename: str) -> List[str]:
    """Get the requirements from a file.

    Args:
        filename (str): The file to get the requirements from.

    Returns:
        List[str]: The requirements.
    """
    line_strings = [dep.strip() for dep in open(filename, "r").readlines()]
    req_strings = []
    use_line = False

    for line_str in line_strings:
        if "dependencies" in line_str:
            use_line = True
            continue
        elif "python" in line_str:
            if "python-json-logger" in line_str:
                use_line = True
            else:
                continue
        elif "pytorch-cpu" in line_str:
            line_str = line_str.replace("pytorch-cpu", "torch")
        elif "matplotlib-base" in line_str:
            line_str = line_str.replace("matplotlib-base", "matplotlib")
        elif "platforms" in line_str:
            use_line = False

        if use_line:
            req_strings.append(line_str[2:])

    return req_strings


setup_dict = dict(
    name="facetorch",
    version=read_version("./version"),
    description="Face analysis PyTorch framework.",
    author="Tomas Gajarsky",
    author_email="gajarsky.tomas@gmail.com",
    maintainer="Tomas Gajarsky",
    maintainer_email="gajarsky.tomas@gmail.com",
    url="https://github.com/tomas-gajarsky/facetorch",
    download_url="https://pypi.org/project/facetorch/",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.9",
    install_requires=get_requirements("environment.yml"),
    tests_require=get_requirements("environment.yml") + ["pytest", "pytest-cov"],
    setup_requires=["wheel", "setuptools>=63.4.2"],
    zip_safe=False,
    license_files=("LICENSE",),
)


def main() -> None:
    setup(**setup_dict)


if __name__ == "__main__":
    main()
