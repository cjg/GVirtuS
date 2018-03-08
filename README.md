# Introducing GVirtuS
GVirtuS is the general virtualization system developed in 2009 and firstly introduced in 2010 enabling a completely transparent layer among GPUs and VMs. GVirtuS has been chosen by the RAPID H2020 project as cornerstone for building the GPU Bridger.

Contacts:

Raffaele Montella

raffaele.montella@uniparthenope.it

Department of Science and Technologies

University of Napoli Parthenope - Italy

# Acknowledge
This has been supported by the European Union Grant Agreement number 644312-RAPID--H2020-ICT-2014/H2020-ICT-2014-1 "Heterogeneous Secure Multi-level Remote Acceleration Service for Low-Power Integrated Systems and Devices" http://rapid-project.eu

# How To install GVirtuS framework and plugins#
## Prerequisites: ##
GCC, G++

OS: Ubuntu 10.04 (tested) /  Ubuntu  12.04 (tested) / Ubuntu 14.04 (tested) / Ubuntu 16.04 (tested) /CentOS 6.8 (tested)

CUDA Toolkit: up to Version 9.1

This package are required:
    build-essential
    autotools-dev
    automake
    git
    libtool
    libxmu-dev
    libxi-dev
    libgl-dev
    libosmesa-dev
    liblog4cplus-dev
    libzmq-dev

Ubuntu:
sudo apt-get install build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev git liblog4cplus-dev libzmq-dev

## Installation: ##
1) Clone the GVirtuS main repository

    git clone https://github.com/raffmont/GVirtuS.git

In the directory “GVirtuS” there are three directories named “gvirtus”, “gvirtus.cudart” and "gvirtus.cudadr".

“gvirtus” contains the framework.

“gvirtus.cudart” and "gvirtus.cudadr" contains the cuda runtime plugin and the cuda driver plugin.


2) Launch the installer script indicating the destination folder of the installation (es. "/home/rapid/opt"):

    ./gvirtus-installer "GVIRTUS_PATH"

To check your installation please check the following directories:

Check GVIRTUS_PATH/lib for frontend and backend directories

## Testing ##

How GVirtuS works? Could my application work with GVirtuS?

The only way to answer is tring and testing.

A good starting poin is testing GVirtuS aganist the NVIDIA CUDA SAMPLES using the gvirtus-test script.

    ./gvitus-test NVIDA_CUDA_SAMPLES_PATH output_file [avoid_file]

Where:

    NVIDA_CUDA_SAMPLES_PATH is the absolute path of the sample files shipped with the NVIDIA CUDA SdK

    output_file is the CSV file where the results will be saved

    avoid_file is a text file containing the samples that have to be skipped in order to avoid the script stucks.

The script produces a CSV files with reluts.

On the standard output the script prints the error message generated by the sample. This could be used for debugging or in order to improve features.

Example:

./gvirtus-test $HOME/dev/cuda-samples/NVIDIA_CUDA-9.0_Samples/ $HOME/dev/GVirtuS-9.0/saturn.saturn.9.0.csv $HOME/dev/GVirtuS-9.0/avoid_file.txt 2>&1 | tee result.txt

## EXAMPLE cuda application ##

### Backend machine (physical GPU and Cuda required) ###

On the remote machine where the cuda executables will be executed

Modify the Gvirtus configuration file backend if the default port 9998 is occuped or the machine is remote:

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
    
    communicator : tcp://127.0.0.1:9998 #change 127.0.0.1 with remote host if necessary
    plugins : cudart, cudadr
    
    #
    # End Of File
    #


Export the dynamic CUDA library:(typically /usr/local/cuda/lib64)


    export LD_LIBRARY_PATH=”<CUDA_PATH>/lib64” 

Execute application server gvirtus-backend with follow command:

    GVIRTUS_PATH/bin/gvirtus-backend

### Frontend machine (No GPU or Cuda required) ###

Modify the Gvirtus configuration file frontend:

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
    
    communicator : tcp://127.0.0.1:9998 #change 127.0.0.1 with the host of backend machine if necessary
    plugins : cudart, cudadr
    
    #
    # End Of File
    #

**NOTE: In the local configuration GvirtuS Backend and Frontend share the same configuration files.**

Export the dynamic GVirtuS library:

    export  LD_LIBRARY_PATH=GVIRTUS_PATH/gvirtus/lib/frontend

Optionally set a different configuration file

    export CONFIG_FILE=$HOME/dev/gvirtus.properties

execute the cuda application compiled with cuda dynamic library (with -lcuda -lcudart)

    ./example

If you are using nvcc be sure you are compiling using shared libraries:

    export EXTRA_NVCCFLAGS="--cudart=shared"


