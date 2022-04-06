from setuptools import setup, find_packages

packages = find_packages()
print('packages: %s' % packages)

setup(name="DM_solver",
        version="3",
        packages = find_packages(),
        )
