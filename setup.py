from setuptools import setup, find_packages


def get_requirements(filename):
    return open(filename).read().strip().split("\n")


REQUIREMENTS = get_requirements("requirements.txt")
TEST_REQUIREMENTS = get_requirements("test-requirements.txt")

setup(
    name="deep_rl",
    description="Deep Reinforcement Learning examples from Udacity's Deep RL Nano "
    "Degree",
    url="https://github.com/daniel-m-campos/deep_rl",
    author="Daniel Campos",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIREMENTS + TEST_REQUIREMENTS,
    extras_require={"TEST": TEST_REQUIREMENTS},
)
