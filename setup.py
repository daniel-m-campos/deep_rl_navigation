from setuptools import setup, find_packages


def get_requirements(filename):
    return open(filename).read().strip().split("\n")


REQUIREMENTS = get_requirements("requirements.txt")
TEST_REQUIREMENTS = get_requirements("test-requirements.txt")

setup(
    name="deep_rl_navigation",
    description="Solution for Udacity Deep Reinforcement Learning Navigation Project",
    url="https://github.com/daniel-m-campos/deep_rl_navigation",
    author="Daniel Campos",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIREMENTS + TEST_REQUIREMENTS,
    extras_require={"TEST": TEST_REQUIREMENTS},
)
