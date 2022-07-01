from setuptools import setup, find_packages
 
 
 
setup(name='optilab',
 
      version='0.1',
 
      url='https://github.com/NekliudovAV/OptiLab',
 
      license='MIT',
 
      author='Nekliudov Aleksei',
 
      author_email='greenocean@yandex.com',
 
      description='Thermal Power Plant Optimisation',
 
      packages=find_packages(exclude=['tests']),
 
      long_description=open('README.md').read(),
 
      zip_safe=False,
 
      setup_requires=['nose>=1.0'],
 
      test_suite='nose.collector')
