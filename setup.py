from setuptools import setup, find_packages


setup(
    name="ScribeTokens",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.13",
)