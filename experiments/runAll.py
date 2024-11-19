import os

experiments_dir = [
      "MountainCarContinuous",
      # "MountainCar",
      "Pendulum",
      # "cart_pole",
      # "Acrobot",
      # "LunarLander",
      # "BipedalWalker",
]

keyWords = [
      "cmaes",
      "large",
      "latent",
      "raw",   
]

# get the current working directory
cwd = os.getcwd()

for experiment in experiments_dir:
   # change the directory to the experiment
   os.chdir(os.path.join(cwd, experiment))
   # list of files in the experiment directory
   files = os.listdir()
   # iterate over the files in the experiment directory
   for file in files:
      # check if the file is a python file and contains the keyword
      if file.endswith(".py") and any(keyword in file for keyword in keyWords):
         # run the python file
         os.system(f"python {file}")
         