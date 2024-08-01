A global trajectory planning algorithm for a mobile robot moving in heterogeneous environments. 

As input data characterizing the surface on which the robot is moving, we use image of a 
polygon (1.jpg)

All algorithms are tuned to find trajectories that have lower energy consumption compared to others.

Algorithms used: Dijkstra, RRT, A*

The obtained images and data are sorted in a folder (AMPS_Full)

Image preprocessing -> CV_Module.py

Neurophysical model -> NeurophysicalModel_Module.py

Mathematical transformations -> Trans_Module.py

Basic algorithms -> Global_Path_Planner.py
