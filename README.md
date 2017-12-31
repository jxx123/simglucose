# simglucose
A Type-1 Diabetes simulator implemented in Python for Reinforcement Learning purpose

This simulator is a python implementation of UVa/Padova Simulator for research purpose only. The simulator includes 30 virtual patients, 10 adolescents, 10 adults, 10 children. 
Main Features:
- Simulation enviroment follows OpenAI gym and rllab APIs. It returns observation, reward, done, info at each step. It is "reinforcement-learning-ready".
- Supports parallel computing. The simulator uses pathos multiprocessing package.
