from setuptools import setup


setup(name='simglucose',
      version='0.1.9',
      description='A Type-1 Diabetes Simulator as a Reinforcement Learning Environment in OpenAI gym or rllab (python implementation of UVa/Padova Simulator)',
      url='https://github.com/jxx123/simglucose',
      author='Jinyu Xie',
      author_email='xjygr08@gmail.com',
      license='MIT',
      packages=['simglucose'],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'matplotlib',
          'pathos',
          'gym==0.9.4'
      ],
      include_package_data=True,
      zip_safe=False)
