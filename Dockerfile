#To build this docker file
#docker build -t gawler .

#To run this, with an interactive python3 temrinal, mounting your current host directory in the container directory at /workspace use:
#docker run -it -p 8888:8888 -v C:\PYG\:/workspace gawler /bin/bash
#jupyter notebook --allow-root --ip=0.0.0.0 --no-browser

# Pull base image.
FROM ubuntu:bionic
MAINTAINER Nathaniel Butterworth

RUN apt-get update -y && \
	apt-get install -y libgdal-dev python3-pip python3-dev python3-gdal gdal-bin
	
#Create some directories to work with
WORKDIR /build
RUN mkdir /workspace 

#Install python libraries
RUN pip3 install shapely --no-binary shapely
RUN pip3 install scipy==1.2 scikit-learn==0.23 matplotlib==3.0 pyshp numpy==1.16 jupyter==1.0 pandas==1.0 notebook==6.0.3 Pillow==7.1.2

RUN pip3 install "dask[complete]"
RUN pip3 install cython 
RUN pip3 install cartopy
RUN pip3 install progress
RUN pip3 install pyevtk
RUN pip3 install swifter

#Make the workspace persistant
VOLUME /workspace
WORKDIR /workspace




