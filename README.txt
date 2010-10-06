
This is the distribution of the Python package cuda-sim.
cuda-sim enables biochemical network simulation on NVIDIA
CUDA GPUSs.


#################################
# CONTENTS 
#################################
1) Prerequisites
2) Linux installation
3) Mac OSX installation

#################################
# 1) Prerequisites
#################################


Before trying to install and use cuda-sim you must
first install
      
      CUDA toolkit

and the following python packages

       numpy
       pycuda 
       libSBML (SBML interface)
	
While CUDA, numpy and pycuda are essential, the libsbml need only
be installed if model import from SBML is required.


#################################
# 2) LINUX installation
#################################

1) CUDA toolkit
Instructions are available here
http://developer.download.nvidia.com/compute/cuda/3_1/docs/GettingStartedLinux.pdf
and the required files are here
http://developer.nvidia.com/object/cuda_3_1_downloads.html#Linux

2) Python (if not already installed)
http://www.python.org/ftp/python/2.6.5/Python-2.6.5.tgz
	
	tar xzf Python-2.6.5.tgz
	cd Python-2.6.5
	./configure --prefix=<dir> --enable-shared
	make
	sudo make install

If custom installation is required then replace <dir> 
with the full path to a location. This will be the 
location containing lib and bin directories (usually 
/usr/local by default).

The --prefix=<dir> option is recommended since it will 
guarantee that each package picks up the correct dependency.

Make sure this new version of python is picked up by default
and this can be added to your .bash_profile or .bashrc etc

	export PATH=<dir>/bin:$PATH  (for BASH)
	setenv PATH <dir>/bin:$PATH

3) Follow the instructions for numpy and pycuda installation
http://wiki.tiker.net/PyCuda/Installation/Linux
This may require installation of boost, distribute and pytools

4) libSBML
Download and install swig 
Note that this is required by libsbml and it must be at least version 1.3.39 
http://downloads.sourceforge.net/project/swig/swig/swig-1.3.40/swig-1.3.40.tar.gz

	tar -xzf swig-1.3.40.tar.gz
	cd swig-1.3.40
	./configure --prefix=<dir>
	make
	sudo make install

Download and install libSBML
http://downloads.sourceforge.net/project/sbml/libsbml/4.0.1/libsbml-4.0.1-src.zip

	unzip libsbml-4.0.1-src.zip
	cd libsbml-4.0.1	
	./configure --with-python=<dir> --prefix=<dir> --with-swig=<dir>
	make 
	sudo make install

5) Download and install cuda-sim
https://sourceforge.net/projects/cuda-sim/files/latest
In the unzipped cuda-sim package directory do
           
	tar xzf cuda-sim-VERSION.tar.gz
	cd cuda-sim-VERSION
	python setup.py install

6) Done!
You should be able to do
	
	python 
	import libsbml
	import cudasim



#################################
# 3) Mac OSX installation
#################################

Notes:
* Installing the dependencies on Mac is more problematic
* 10.5 should work ok with the following instructions 
* 10.6 is more difficult. More details are provided in the 
  pycuda instructions

1) CUDA toolkit 3.1 is recommended
Instructions are available here
http://developer.download.nvidia.com/compute/cuda/3_1/docs/GettingStartedMacOS.pdf
and the required files are here
http://developer.nvidia.com/object/cuda_3_1_downloads.html#MacOS

Notes:
When running the device query do you get 'NO CUDA DEVICE FOUND Device Emulation (CPU)'
Try this fix and restarting:
	 sudo chmod 755 /System/Library/StartupItems/CUDA/*
	 sudo chmod 755 /System/Library/StartupItems/CUDA

2) Follow the instructions for numpy and pycuda installation
http://wiki.tiker.net/PyCuda/Installation/Mac
This may require installation of boost, distribute and pytools

Download and install BOOST
http://sourceforge.net/projects/boost/files/boost/1.39.0/boost_1_39_0.tar.gz/download
On 10.5 it suffices to do:

	tar -xzf boost_1_39_0.tar.gz
	cd boost_1_39_0
	./bootstrap.sh --prefix=/usr/local/ --libdir=/usr/local/lib --with-libraries=signals,thread,python \
		       --with-python=/Library/Frameworks/Python.framework/Versions/2.6/Python
	./bjam variant=release link=shared install		
	export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH   #(ADD THIS TO .bash_profile)

Download and install numpy
http://downloads.sourceforge.net/project/numpy/NumPy/1.4.1rc2/numpy-1.4.1rc2-py2.6-python.org.dmg 

Download and install pycuda
http://pypi.python.org/packages/source/p/pycuda/pycuda-0.94rc.tar.gz
Note you will need to look at which version of gcc is used. On my system it was 4.0.
Look at the boost libraries (ie /usr/local/lib/libboost*) to see what extensions they have	
	
	tar xzf pycuda-0.94rc.tar.gz
	cd pycuda-0.94rc
	python configure.py --boost-inc-dir=/usr/local/include/boost-1_39 --boost-lib-dir=/usr/local/lib \
	       --boost-python-libname=boost_python-xgcc40-mt --boost-thread-libname=boost_thread-xgcc40-mt \
	       --cuda-root=/usr/local/cuda --boost-compiler=gcc40
	sudo make install	
	
Perform the tests
	cd pycuda-0.94rc/test
	python test_driver.py

If you have trouble, edit the siteconf.py file and retry the 'sudo make install'. 
For example on my 10.5 system it looked like this

BOOST_INC_DIR = ['/usr/local/include/boost-1_39']
BOOST_LIB_DIR = ['/usr/local/lib']
BOOST_COMPILER = 'gcc40'
BOOST_PYTHON_LIBNAME = ['boost_python-xgcc40-mt']
BOOST_THREAD_LIBNAME = ['boost_thread-xgcc40-mt']
CUDA_TRACE = False
CUDA_ROOT = '/usr/local/cuda'
CUDA_ENABLE_GL = False
CUDADRV_LIB_DIR = []
CUDADRV_LIBNAME = ['cuda']
CXXFLAGS = []
LDFLAGS = []

CXXFLAGS.extend(['-isysroot', '/Developer/SDKs/MacOSX10.5.sdk'])
LDFLAGS.extend(['-isysroot', '/Developer/SDKs/MacOSX10.5.sdk'])


4) libSBML
Download and install libSBML
http://downloads.sourceforge.net/project/sbml/libsbml/4.0.1/libsbml-4.0.1-src.zip

	unzip libsbml-4.0.1-src.zip
	cd libsbml-4.0.1	
	./configure --with-python=/usr/local
	make 
	sudo make install
	
This installs into the wrong directory, so we need to move it to the correct place
cp -r /usr/local/lib/python2.6/site-packages/libsbml* /Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/

5) Download and install cuda-sim	
https://sourceforge.net/projects/cuda-sim/files/latest

	tar xzf cuda-sim-VERSION.tar.gz
	cd cuda-sim-VERSION
	python setup.py install

This places the cudasim package into 
     
	/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/

6) Done!
You should be able to do
	
	python 
	import libsbml
	import cudasim

