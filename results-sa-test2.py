#!/usr/bin/env python
# coding: utf-8

# Used to run the poin-in-polygon co-registrations
# as these can take a looong time (20 hours for 1,000,000 points)
# Uses swifter to parallelise.
#Run as final step to produce test/target data set.


#Import libraries for data manipulations
import pandas as pd
import numpy as np

#Import libraries for tif, shapefile, and geodata manipulations
import shapefile

#Import libraries for multi-threading capabilities
#from dask import delayed,compute
#from dask.distributed import Client, progress
import time
import swifter

# The very slow algorithm to find the what polygon your point is in.
from shapely.geometry import Point
from shapely.geometry import shape

def shapeExplore(point,shapes,recs,record):
    #record is the column index you want returned
    for i in range(len(shapes)):
        boundary = shapes[i]
        if Point((point.lon,point.lat)).within(shape(boundary)):
            return(recs[i][record])
    #if you have been through the loop with no result
    return(-9999.)


#Load in only the datasets we need.
# #Categorised geology
geolshape=shapefile.Reader("shapes/geology_simp.shp")

recsGeol    = geolshape.records()
shapesGeol  = geolshape.shapes()

geolshape=shapefile.Reader("shapes/Archaean - Early Mesoproterozoic polygons.shp")

recsArch   = geolshape.records()
shapesArch  = geolshape.shapes()


# # Part 2 - Spatial data mining of datasets

#Load in the mostly co-registered dataframe and add the geology
target_data=pd.read_csv("target_data_01nogeol.csv",header=0)

#from dask import dataframe as dd 
#sd = dd.from_pandas(target_data,npartitions=20)
tic=time.time()
#Add the categorical shapefile data
target_data['geol28']=target_data.swifter.apply(shapeExplore, args=(shapesGeol,recsGeol,1), axis=1)
target_data['archean27']=target_data.swifter.apply(shapeExplore, args=(shapesArch,recsArch,-1), axis=1)
toc=time.time()
print("Time taken geol:", toc-tic, " seconds")


#Save out the final data-mined co-registered dataset to be used for ML test/targeting.
target_data.to_csv("target_data.csv",index=False)





