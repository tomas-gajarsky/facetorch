import codecs
import os.path
from distutils.core import setup
from typing import List

from setuptools import find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


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
        elif "pytorch" in line_str:
            line_str = line_str[2:]
        elif "platforms" in line_str:
            use_line = False

        if use_line:
            req_strings.append(line_str[2:])

    return req_strings


setup_dict = dict(
    name="facetorch",
    version=get_version("facetorch/__init__.py"),
    author="Tomas Gajarsky",
    author_email="gajarsky.tomas@gmail.com",
    maintainer="Tomas Gajarsky",
    maintainer_email="gajarsky.tomas@gmail.com",
    url="https://github.com/tomas-gajarsky/facetorch",
    description="Face analysis PyTorch framework.",
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
    tests_require=get_requirements("environment.yml"),
    zip_safe=False,
    license_files=("LICENSE.txt",),
)


def main() -> None:
    setup(**setup_dict)


if __name__ == "__main__":
    main()
