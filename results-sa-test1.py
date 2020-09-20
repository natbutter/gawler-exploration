#!/usr/bin/env python
# coding: utf-8

#This script generates the coregistered test set for the Gawler region.
#Use with the results-sa.ipynb and with results-sa-test2.py

#Import libraries for data manipulations
import pandas as pd
import numpy as np
import random
import scipy
from scipy import io

#Import libraries for plotting
#import matplotlib.pyplot as plt
#import matplotlib.ticker as mticker
#import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits import mplot3d
import matplotlib.mlab as ml
#from cartopy.io.img_tiles import Stamen
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch

#Import libraries for tif, shapefile, and geodata manipulations
import shapefile

#Import Machine Learning libraries
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Import libraries for multi-threading capabilities
from dask import delayed,compute
from dask.distributed import Client, progress
import time

#Define functions
def coregPoint(point,data,region):
    '''
    Finds the nearest neighbour to a point from a bunch of other points
    point - array([longitude,latitude])
    data - array
    region - integer, same units as data
    '''
    tree = scipy.spatial.cKDTree(data)
    dists, indexes = tree.query(point,k=1,distance_upper_bound=region) 

    if indexes==len(data):
        return 'inf'
    else:
        return (indexes,dists)
    
    
def points_in_circle(circle, arr):
    '''
    A generator to return all points whose indices are within given circle.
    http://stackoverflow.com/a/2774284
    Warning: If a point is near the the edges of the raster it will not loop 
    around to the other side of the raster!
    '''
    i0,j0,r = circle

    for i in range(intceil(i0-r),intceil(i0+r)):
        ri = np.sqrt(r**2-(i-i0)**2)
        for j in range(intceil(j0-ri),intceil(j0+ri)):
            if (i >= 0 and i < len(arr[:,0])) and (j>=0 and j < len(arr[0,:])):
                yield arr[i][j]
            
def intceil(x):
    return int(np.ceil(x))                                            


def coregRaster(point,data,region):
    '''
    Finds the mean value of a raster, around a point with a specified radius.
    point - array([longitude,latitude])
    data - array
    region - integer, same units as data
    '''
    i0=point[1]
    j0=point[0]
    r=region #In units of degrees
    pts_iterator = points_in_circle((i0,j0,region), data)
    pts = np.array(list(pts_iterator))
    #remove values outside the region which for there is no data (0.0).
    #print(pts)
    pts = pts[pts != 0.]
    if np.isnan(np.nanmean(pts)):
        #print(point,"nan")
        #pts=np.median(data)
        pts=-9999.
        #print("returning",pts)

    #return(scipy.stats.nanmean(pts)) #deprecated from scipy 0.15
    return(np.nanmean(pts))


#Make a function that can turn point arrays into a full meshgrid
def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


# # Part 1 
# ### Wrangling the raw data
# 
# ### Load in Deposit locations - mine and mineral occurances

#Set the filename
mineshape="shapes/mines_and_mineral_occurrences_all.shp"

#Set shapefile attributes and assign
sf = shapefile.Reader(mineshape)
fields = [x[0] for x in sf.fields][1:]
records = sf.records()
shps = [s.points for s in sf.shapes()]

#write into a dataframe
df = pd.DataFrame(columns=fields, data=records)

#Get the gawler map boundary
mineshape="shapes/GCAS_Boundary.shp"

#read in the file
shapeRead = shapefile.Reader(mineshape)
#And save out some of the shape file attributes
shapes  = shapeRead.shapes()
xval = [x[0] for x in shapes[0].points]
yval = [x[1] for x in shapes[0].points]


commname='Cu'
comm=df[df['COMM_CODE'].str.contains(commname)]
comm=comm.reset_index(drop=True)
print(comm.shape)

commsig=comm[comm.SIZE_VAL!="Low Significance"]

# # Wrangle the geophysical and geological datasets
# Each geophysical dataset could offer instight into various commodities. 
# Here we load in the pre-formatted datasets and prepare them for further manipulations. 

# ## Resistivity
data_res=pd.read_csv("sa/AusLAMP_MT_Gawler.xyzr",
                     sep='\s+',header=0,names=['lat','lon','depth','resistivity'])

lon_res=data_res.lon.values
lat_res=data_res.lat.values
depth_res=data_res.depth.values
res_res=data_res.resistivity.values

f=[]
for i in data_res.depth.unique():
    f.append(data_res[data_res.depth==i].values)

f=np.array(f)
print("Resitivity in:", np.shape(f))

#Set an array we can interrogate values of later
#This is the same for all resistivity vectors
lonlatres=np.c_[f[0,:,1],f[0,:,0]]
lonres=f[0,:,1]
latres=f[0,:,0]


# ## Faults and dykes

#Get fault data neo
faultshape="shapes/Neoproterozoic - Ordovician faults.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsNeo=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsNeo.append([j[0],j[1]])
faultsNeo=np.array(faultsNeo)

#Get fault data archean
faultshape="shapes/Archaean - Early Mesoproterozoic faults.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsArch=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsArch.append([j[0],j[1]])
faultsArch=np.array(faultsArch)

#Get fault data dolerite dykes swarms
faultshape="shapes/Gairdner Dolerite.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsGair=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsGair.append([j[0],j[1]])
faultsGair=np.array(faultsGair)


# ### Netcdf formatted
#Define a function to read netcdf data
def readnc(filename):
    tic=time.time()
    rasterfile=filename
    data = scipy.io.netcdf_file(rasterfile,'r')
    xdata=data.variables['lon'][:]
    ydata=data.variables['lat'][:]
    zdata=np.array(data.variables['Band1'][:])

    toc=time.time()
    print(rasterfile, "in", toc-tic)
    print("spacing x", xdata[2]-xdata[1], "y", ydata[2]-ydata[1], np.shape(zdata),np.min(xdata),np.max(xdata),np.min(ydata),np.max(ydata))

    return(xdata,ydata,zdata)

#Load in the grids
x1,y1,z1 = readnc("sa/aster-AlOH-cont.nc")
x2,y2,z2 = readnc("sa/aster-AlOH-comp.nc")
x3,y3,z3 = readnc("sa/aster-FeOH-cont.nc")
x4,y4,z4 = readnc("sa/aster-Ferric-cont.nc")
x5,y5,z5 = readnc("sa/aster-Ferrous-cont.nc")
x6,y6,z6 = readnc("sa/aster-Ferrous-index.nc")
x7,y7,z7 = readnc("sa/aster-MgOH-comp.nc")
x8,y8,z8 = readnc("sa/aster-MgOH-cont.nc")
x9,y9,z9 = readnc("sa/aster-green.nc")
x10,y10,z10 = readnc("sa/aster-kaolin.nc")
x11,y11,z11 = readnc("sa/aster-opaque.nc")
x12,y12,z12 = readnc("sa/aster-quartz.nc")
x13,y13,z13 = readnc("sa/aster-regolith-b3.nc")
x14,y14,z14 = readnc("sa/aster-regolith-b4.nc")
x15,y15,z15 = readnc("sa/aster-silica.nc")
x16,y16,z16 = readnc("sa/sa-base-elev.nc")
x17,y17,z17 = readnc("sa/sa-dem.nc")
x18,y18,z18 = readnc("sa/sa-base-dtb.nc")
x19,y19,z19 = readnc("sa/sa-mag-2vd.nc")
x20,y20,z20 = readnc("sa/sa-mag-rtp.nc")
x21,y21,z21 = readnc("sa/sa-mag-tmi.nc")
x22,y22,z22 = readnc("sa/sa-rad-dose.nc")
x23,y23,z23 = readnc("sa/sa-rad-k.nc")
x24,y24,z24 = readnc("sa/sa-rad-th.nc")
x25,y25,z25 = readnc("sa/sa-rad-u.nc")
x26,y26,z26 = readnc("sa/sa-grav.nc")


#Define a function to find points in polygons
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

# #Categorised geology
geolshape=shapefile.Reader("shapes/geology_simp.shp")

recsGeol    = geolshape.records()
shapesGeol  = geolshape.shapes()

geolshape=shapefile.Reader("shapes/Archaean - Early Mesoproterozoic polygons.shp")

recsArch   = geolshape.records()
shapesArch  = geolshape.shapes()


# # Part 2 - Spatial data mining of datasets
# 
# ### Select the commodity and geophysical features to use 
# (edit *commname* above and turn feature labels on/off here as required)

lons=['lon','lat']
reslabels = [     
'res-25',
'res-77',
'res-136',
'res-201',
'res-273',
'res-353',
'res-442',
'res-541',
'res-650',  
'res-772',
'res-907',
'res-1056',
'res-1223',
'res-1407',
'res-1612',
'res-1839',
'res-2092',
'res-2372',
'res-2683',
'res-3028',
'res-3411',
'res-3837',    
'res-4309',
'res-4833',
'res-5414',
'res-6060',
'res-6776',
'res-7572',
'res-8455',
'res-9435',
'res-10523',
'res-11730',
'res-13071',
'res-14559',
'res-16210',
'res-18043',   
'res-20078',
'res-22337',
'res-24844',
'res-27627',
'res-30716',
'res-34145',
'res-37951',
'res-42175',
'res-46865',
'res-52070',
'res-57847',
'res-64261',
'res-71379',
'res-79281',
'res-88052',
'res-97788',
'res-108595',
'res-120590',
'res-133905',
'res-148685',
'res-165090',
'res-183300',
'res-203513',
'res-225950',
'res-250854',
'res-278498',
'res-309183'
]
  
faultlabels=[
    "neoFaults",
    "archFaults",
    "gairFaults"
]
   
numerical_features=reslabels+faultlabels+[
"aster1-AlOH-cont",
"aster2-AlOH",
"aster3-FeOH-cont",
"aster4-Ferric-cont",
"aster5-Ferrous-cont",
"aster6-Ferrous-index",
"aster7-MgOH-comp",
"aster8-MgOH-cont",
"aster9-green",
"aster10-kaolin",
"aster11-opaque",
"aster12-quartz",
"aster13-regolith-b3",
"aster14-regolith-b4",
"aster15-silica",
"base16",
"dem17",
"dtb18",
"mag19-2vd",
"mag20-rtp",
"mag21-tmi",
"rad22-dose",
"rad23-k",
"rad24-th",
"rad25-u",
"grav26"
]

categorical_features=[
'archean27',
'geol28',
'random'
]



# #Generate "non-deposit points on land (or in the gawler) for sa"
polgonshape=shapefile.Reader("shapes/SA_STATE_POLYGON_shp.shp")
# #polgonshape=shapefile.Reader("/workspace/DATA/RAW/zips/Unearthed_5_GCAS_Boundary/GCAS_Boundary.shp")

shapesPoly  = polgonshape.shapes()


#We may want to train and test just over the regions that the grids are valid.
#So we can crop the known deposits to the extent of the grids.
#comm=comm[(comm.lon<max(xval)) & (comm.lon>min(xval)) & (comm.lat>min(yval)) & (comm.lat<max(yval))]

sizes=np.shape(comm)
print(sizes)
#Now make a set of "non-deposits" using a random location within our exploration area
lats_rand=np.random.uniform(low=min(df.LATITUDE), high=max(df.LATITUDE), size=len(comm.LATITUDE))
lons_rand=np.random.uniform(low=min(df.LONGITUDE), high=max(df.LONGITUDE), size=len(comm.LONGITUDE))

#And enforce the random points are on the land
boundary=shapesPoly[1]
for i,_ in enumerate(lats_rand):
    while not Point((lons_rand[i],lats_rand[i])).within(shape(boundary)):
            lats_rand[i]=random.uniform(min(df.LATITUDE), max(df.LATITUDE))
            lons_rand[i]=random.uniform(min(df.LONGITUDE), max(df.LONGITUDE))

xvalsa = [x[0] for x in shapesPoly[1].points]
yvalsa = [x[1] for x in shapesPoly[1].points]

#Define a function to coregister the grids. 
#Requires list of lat and lon, will return the value at that point for all the hardcoded grids
#Hadrcoded grids currently defined globally, todo
def coregLoop(sampleData):
    
    lat=sampleData[0]
    lon=sampleData[1]
    region=1 #degrees
    region2=100

    #Resitivity indexes
    idx,dist=coregPoint([lon,lat],lonlatres,region2)
    
    #faultdist
    _,dist=coregPoint([lon,lat],faultsNeo,region2)
    _,dist2=coregPoint([lon,lat],faultsArch,region2)
    _,dist3=coregPoint([lon,lat],faultsGair,region2)

    #Numerical data indexes
    xloc1=(np.abs(np.array(x1) - lon).argmin())
    yloc1=(np.abs(np.array(y1) - lat).argmin())
    xloc2=(np.abs(np.array(x2) - lon).argmin())
    yloc2=(np.abs(np.array(y2) - lat).argmin())
    xloc3=(np.abs(np.array(x3) - lon).argmin())
    yloc3=(np.abs(np.array(y3) - lat).argmin())
    xloc4=(np.abs(np.array(x4) - lon).argmin())
    yloc4=(np.abs(np.array(y4) - lat).argmin())
    xloc5=(np.abs(np.array(x5) - lon).argmin())
    yloc5=(np.abs(np.array(y5) - lat).argmin())
    xloc6=(np.abs(np.array(x6) - lon).argmin())
    yloc6=(np.abs(np.array(y6) - lat).argmin())
    xloc7=(np.abs(np.array(x7) - lon).argmin())
    yloc7=(np.abs(np.array(y7) - lat).argmin())
    xloc8=(np.abs(np.array(x8) - lon).argmin())
    yloc8=(np.abs(np.array(y8) - lat).argmin())
    xloc9=(np.abs(np.array(x9) - lon).argmin())
    yloc9=(np.abs(np.array(y9) - lat).argmin())
    xloc10=(np.abs(np.array(x10) - lon).argmin())
    yloc10=(np.abs(np.array(y10) - lat).argmin())
    xloc11=(np.abs(np.array(x11) - lon).argmin())
    yloc11=(np.abs(np.array(y11) - lat).argmin())
    xloc12=(np.abs(np.array(x12) - lon).argmin())
    yloc12=(np.abs(np.array(y12) - lat).argmin())
    xloc13=(np.abs(np.array(x13) - lon).argmin())
    yloc13=(np.abs(np.array(y13) - lat).argmin())
    xloc14=(np.abs(np.array(x14) - lon).argmin())
    yloc14=(np.abs(np.array(y14) - lat).argmin())
    xloc15=(np.abs(np.array(x15) - lon).argmin())
    yloc15=(np.abs(np.array(y15) - lat).argmin())
    xloc16=(np.abs(np.array(x16) - lon).argmin())
    yloc16=(np.abs(np.array(y16) - lat).argmin())
    xloc17=(np.abs(np.array(x17) - lon).argmin())
    yloc17=(np.abs(np.array(y17) - lat).argmin())
    xloc18=(np.abs(np.array(x18) - lon).argmin())
    yloc18=(np.abs(np.array(y18) - lat).argmin())
    xloc19=(np.abs(np.array(x19) - lon).argmin())
    yloc19=(np.abs(np.array(y19) - lat).argmin())
    xloc20=(np.abs(np.array(x20) - lon).argmin())
    yloc20=(np.abs(np.array(y20) - lat).argmin())
    xloc21=(np.abs(np.array(x21) - lon).argmin())
    yloc21=(np.abs(np.array(y21) - lat).argmin())
    xloc22=(np.abs(np.array(x22) - lon).argmin())
    yloc22=(np.abs(np.array(y22) - lat).argmin())
    xloc23=(np.abs(np.array(x23) - lon).argmin())
    yloc23=(np.abs(np.array(y23) - lat).argmin())
    xloc24=(np.abs(np.array(x24) - lon).argmin())
    yloc24=(np.abs(np.array(y24) - lat).argmin())
    xloc25=(np.abs(np.array(x25) - lon).argmin())
    yloc25=(np.abs(np.array(y25) - lat).argmin())
    xloc26=(np.abs(np.array(x26) - lon).argmin())
    yloc26=(np.abs(np.array(y26) - lat).argmin())

    #Categorical data indexes are done with point in polygon
  
    #Numerical data values
    z1val=coregRaster([xloc1,yloc1],z1,region)
    z2val=coregRaster([xloc2,yloc2],z2,region)
    z3val=coregRaster([xloc3,yloc3],z3,region)
    z4val=coregRaster([xloc4,yloc4],z4,region)
    z5val=coregRaster([xloc5,yloc5],z5,region)
    z6val=coregRaster([xloc6,yloc6],z6,region)
    z7val=coregRaster([xloc7,yloc7],z7,region)
    z8val=coregRaster([xloc8,yloc8],z8,region)
    z9val=coregRaster([xloc9,yloc9],z9,region)
    z10val=coregRaster([xloc10,yloc10],z10,region)
    z11val=coregRaster([xloc11,yloc11],z11,region)
    z12val=coregRaster([xloc12,yloc12],z12,region)
    z13val=coregRaster([xloc13,yloc13],z13,region)
    z14val=coregRaster([xloc14,yloc14],z14,region)
    z15val=coregRaster([xloc15,yloc15],z15,region)
    z16val=coregRaster([xloc16,yloc16],z16,region)
    z17val=coregRaster([xloc17,yloc17],z17,region)
    z18val=coregRaster([xloc18,yloc18],z18,region)
    z19val=coregRaster([xloc19,yloc19],z19,region)
    z20val=coregRaster([xloc20,yloc20],z20,region)
    z21val=coregRaster([xloc21,yloc21],z21,region)
    z22val=coregRaster([xloc22,yloc22],z22,region)
    z23val=coregRaster([xloc23,yloc23],z23,region)
    z24val=coregRaster([xloc24,yloc24],z24,region)
    z25val=coregRaster([xloc25,yloc25],z25,region)
    z26val=coregRaster([xloc26,yloc26],z26,region)
    
    #Append all the values to an array to return
    vals=np.array([lon,lat])
    vals=np.append(vals,f[:,idx,3])
    vals=np.append(vals,
                   [
                    dist,dist2,dist3,
                    z1val,z2val,z3val,
                    z4val,z5val,z6val,
                    z7val,z8val,z9val,
                        z10val,z11val,z12val,
                    z13val,z14val,z15val,
                    z16val,z17val,z18val,
                    z19val,z20val,z21val,
                    z22val,z23val,z24val,
                    z25val,z26val,
                    -9999.,-9999.
                   ])
    coregMap=np.append(vals,[random.choice([-999, 999])])
    
    return(coregMap)
    


# ## Run spatial mining of known deposits and "non-deposits"
# Must be re-run on each commodity change.
## THIS IS PREFERENTIALLY DONE IN THE NOTEBOOK VERSION OF THIS SCRIPT ##

# Load in training data
# # training_data=pd.read_csv("training_data-"+commname+"-sig.csv",header=0)
# Or run the next few lines....


# In[498]:


# Interrogate the data associated with deposits
# # tic=time.time()
# # deps1=[]
# # for row in comm.itertuples():
    # # lazy_result = coregLoop([row.LATITUDE,row.LONGITUDE])
    # lazy_result = delayed(coregLoop)([row.LATITUDE,row.LONGITUDE])
    # # deps1.append(lazy_result)
    
# # vec1=pd.DataFrame(np.squeeze(deps1),columns=lons+numerical_features+categorical_features)
# # vec1['deposit'] = 1 #Add the "depoist category flag"

# # toc=time.time()
# # print("Time deposits:", toc-tic, " seconds")
# # tic=time.time()

# Interrogate the data associated with randomly smapled points to use as counter-examples
# # deps0=[]
# # for lat,lon in zip(lats_rand,lons_rand):
    # # lazy_result = coregLoop([lat,lon])
    # lazy_result = delayed(coregLoop)([lat,lon])
    # # deps0.append(lazy_result)
    
# # vec2=pd.DataFrame(np.squeeze(deps0),columns=lons+numerical_features+categorical_features)
# # vec2['deposit'] = 0 #Add the "non-deposit category flag"

# # toc=time.time()
# # print("Time non-deposits:", toc-tic, " seconds")


# Combine the datasets
# # training_data = pd.concat([vec1, vec2], ignore_index=True)

# # tic=time.time()

# Add the categorical shapefile data
# # training_data['geol28']=training_data.apply(shapeExplore, args=(shapesGeol,recsGeol,1), axis=1)
# # training_data['archean27']=training_data.apply(shapeExplore, args=(shapesArch,recsArch,-1), axis=1)

# # toc=time.time()
# # print("Time geology:", toc-tic, " seconds")

# And save the training data out to a file
# # training_data.to_csv("training_data-"+commname+"-sig.csv")

# Save number of deps/non-deps
# # lennon=len(training_data.deposit[training_data.deposit==0])
# # lendep=len(training_data.deposit[training_data.deposit==1])

# # print(lennon,lendep)
# # training_data


# ## Run spatial mining of gridded data
# Only needs to be done once. Then the values of the grid are used to predict whatever commodity is run.

# Load in target data
# # target_data=pd.read_csv("target_data.csv",header=0)

#Or run the next few lines....

#Make a regularly spaced grid here for use in making a probablilty map later
lats_reg=np.arange(min(yval),max(yval)+0.0100,0.01)
lons_reg=np.arange(min(xval),max(xval)+0.0100,0.01)

sampleData=[]
for lat in lats_reg:
    for lon in lons_reg:
            sampleData.append([lat, lon])

print(np.size(sampleData))

#client = Client()

gridgawler=[]
tic=time.time()
for geophysparams in sampleData:
    lazy_result = delayed(coregLoop)(geophysparams)
    #lazy_result = coregLoop(geophysparams)
    gridgawler.append(lazy_result)
print("appended, now running...")

gridgawler=compute(gridgawler)
toc=time.time()

print("Time taken coreg:", toc-tic, " seconds")

target_data=pd.DataFrame(np.squeeze(gridgawler),columns=lons+numerical_features+categorical_features)

### Because of the resource/time differences between the last function
# and this next one. It has been split into two different python scripts.
#Run this one first, and then that one.

#from dask import dataframe as dd 
#sd = dd.from_pandas(target_data,npartitions=20)
tic=time.time()
#Add the categorical shapefile data
#target_data['geol28']=target_data.apply(shapeExplore, args=(shapesGeol,recsGeol,1), axis=1)
#target_data['archean27']=target_data.apply(shapeExplore, args=(shapesArch,recsArch,-1), axis=1)
toc=time.time()
print("Time taken geol:", toc-tic, " seconds")

##Save out the coregistered dataset (which still needs the geology linked)
target_data.to_csv("target_data_01nogeol.csv",index=False)

#Continue by running results-sa-test2.py




