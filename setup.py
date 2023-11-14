from setuptools import setup

setup(
    name="simglucose",
    version="0.2.9",
    description="A Type-1 Diabetes Simulator as a Reinforcement Learning Environment in OpenAI gym or rllab (python implementation of UVa/Padova Simulator)",
    url="https://github.com/jxx123/simglucose",
    author="Jinyu Xie",
    author_email="xjygr08@gmail.com",
    license="MIT",
    packages=["simglucose"],
    install_requires=[
        "gym==0.9.4",
        "gymnasium~=0.29.1",
        "pathos>=0.3.1",
        "scipy>=1.11.0",
        "matplotlib>=3.7.2",
        "numpy>=1.25.0",
        "pandas>=2.0.3",
    ],
    include_package_data=True,
    zip_safe=False,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
