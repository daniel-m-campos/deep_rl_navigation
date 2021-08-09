from setuptools import setup, find_packages


def get_requirements(filename):
    return open(filename).read().strip().split("\n")


setup(
    name="python_template",
    description="Python packaging template",
    url="https://github.com/daniel-m-campos/python_template",
    author="Daniel Campos",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=get_requirements("requirements.txt"),
    test_requires=get_requirements("test-requirements.txt"),
)
