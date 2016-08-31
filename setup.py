from distutils.core import setup

setup(name='cuda-sim',
      version='0.08',
      description='Biochemical network simulation using NVIDIA CUDA',

      author='Yanxiang Zhou, Chris Barnes',

      author_email='christopher.barnes@imperial.ac.uk',

      url='http://sourceforge.net/projects/cuda-sim/',

      packages=['cudasim', 'cudasim.writers', 'cudasim.solvers', 'cudasim.solvers.python'],

      scripts=[],

      package_data={'cudasim': ['MersenneTwister.dat', 'MersenneTwister.cu', 'cuLsoda_all.cu', 'WarpStandard.cu']},

      requires=['libSBML',
                'Numpy',
                'pycuda']
      )
