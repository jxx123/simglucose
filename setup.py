from setuptools import setup


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]


reqs = parse_requirements("requirements.txt")

setup(
    name="simglucose",
    version="0.2.3",
    description="A Type-1 Diabetes Simulator as a Reinforcement Learning Environment in OpenAI gym or rllab (python implementation of UVa/Padova Simulator)",
    url="https://github.com/jxx123/simglucose",
    author="Jinyu Xie",
    author_email="xjygr08@gmail.com",
    license="MIT",
    packages=["simglucose"],
    install_requires=reqs,
    include_package_data=True,
    zip_safe=False,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
