from setuptools import setup


setup(name='simglucose',
      version='0.1',
      description='A Type-1 Diabetes simulator as an OpenAI gym environment',
      url='',
      author='Jinyu Xie',
      author_email='xjygr08@gmail.com',
      license='MIT',
      packages=['simglucose'],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'matplotlib',
      ],
      include_package_data=True,
      zip_safe=False)
