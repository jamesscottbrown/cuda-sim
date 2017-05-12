import os
from distutils.core import setup
from distutils.command.install import install as _install


# extend the install class
class Install(_install):
    def run(self):
        _install.run(self)
        src_dir = self.install_lib + "cudasim/src"
        print "src_dir", src_dir
        comm = "cd " + src_dir + "; chmod +x install_nm.sh; ./install_nm.sh"
        os.system(comm)


setup(name='cuda-sim',
      version='0.08',
      description='Biochemical network simulation using NVIDIA CUDA',

      author='Yanxiang Zhou, Chris Barnes',

      author_email='christopher.barnes@imperial.ac.uk',

      url='http://sourceforge.net/projects/cuda-sim/',

      packages=['cudasim', 'cudasim.writers', 'cudasim.solvers',
                'cudasim.solvers.python', 'cudasim.solvers.cuda', 'cudasim.solvers.c'],

      scripts=[],

      package_data={'cudasim': ['solvers/cuda/MersenneTwister.dat', 'solvers/cuda/MersenneTwister.cu', 'solvers/cuda/cuLsoda_all.cu', 'solvers/cuda/WarpStandard.cu',
                                'src/*']},

      cmdclass={"install": Install},

      requires=['libSBML',
                'Numpy',
                'pycuda']
      )
