# How To install GVirtuS and gvirtus-cudart plugin#
## Prerequisites: ##
GCC, G++

OS: Ubuntu 10.04 (tested) / 12.04 (tested) / 14.04 (tested) 

CUDA Toolkit: Version 6.5

## Installation: ##
1) Clone the GVirtuS main repository

    hg clone https://bitbucket.org/montella/gvirtus-devel

In the directory “gvirtus-devel” there are two directories named “gvirtus” and “gvirtus.cudart”.

“gvirtus” contains the framework.

“gvirtus.cudart” contains the cuda runtime plugin.

The follow steps are required in both the folders.

2) Generate config files and check dependencies

    ./autogen.sh 

3) Compile and install

    make && make install

This default installation will place GvirtuS in the /usr/local directory, if you wish to change the path you should use  

    ./autogen.sh --prefix=”GVIRTUS_PATH”
     make && make install 

To check your installation please check the following directories (default path without --prefix):

/usr/local/include/gvirtus for headers

/usr/local/lib for libraries

Check /usr/local/lib or GVIRTUS_PATH/lib for frontend and backend directories


## EXAMPLE cuda application ##

### Backend machine (physical GPU and Cuda required) ###

On the remote machine where the cuda executables will be executed

Modify the Gvirtus configuration file backend:

/usr/local/etc/gvirtus.properties or GVIRTUS_PATH/etc/gvirtus.properties

    #
    # gVirtuS config file
    #
    
    #
    # Communicator
    #   AfUnix: afunix://path:mode
    #   Shm: shm://
    #   Tcp: tcp://hostname:port
    #   VMShm: vmshm://hostname:port
    
    communicator : tcp://localhost:9988 #change localhost with remote host if necessary
    plugins : cudart
    
    #
    # End Of File
    #


Export the dynamic CUDA library:(typically /usr/local/cuda/lib64)


    export LD_LIBRARY_PATH=”<CUDA_PATH>/lib64” 

Execute application server gvirtus-backend with follow command:

    /usr/local/bin/gvirtus-backend

or

    GVIRTUS_PATH/bin/gvirtus-backend

### Frontend machine (No GPU or Cuda required) ###

Modify the Gvirtus configuration file frontend:

/usr/local/etc/gvirtus.properties 

or

GVIRTUS_PATH/etc/gvirtus.properties



    #
    # gVirtuS config file
    #
    
    #
    # Communicator
    #   AfUnix: afunix://path:mode
    #   Shm: shm://
    #   Tcp: tcp://hostname:port
    #   VMShm: vmshm://hostname:port
    
    communicator : tcp://localhost:9988 #change localhost with the host of backend machine if necessary
    plugins : cudart
    
    #
    # End Of File
    #

**NOTE: In the local configuration GvirtuS Backend and Frontend share the same configuration files.**

Create a soft link manually in this alpha-version with the follow command:

    ln -s GVIRTUS_PATH/lib/frontend/libcudart.so GVIRTUS_PATH/lib/frontend/libcudart.so.6.5.14

Export the dynamic GVirtuS library:

    export  LD_LIBRARY_PATH=GVIRTUS_PATH/gvirtus/lib/frontend

execute the cuda application compiled with cuda dynamic library (with -lcuda -lcudart)

    ./example