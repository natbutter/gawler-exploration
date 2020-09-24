#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries for data manipulations
import pandas as pd
import numpy as np
import random
import scipy
from scipy import io

#Import libraries for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits import mplot3d
import matplotlib.mlab as ml
from cartopy.io.img_tiles import Stamen
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch

#Import libraries for tif, shapefile, and geodata manipulations
#from osgeo import gdal, osr
import shapefile

#from Utils_coreg import *

#Import Machine Learning libraries
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.svm import SVC
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


# In[2]:


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
    
#
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

#            
def intceil(x):
    return int(np.ceil(x))                                            

#
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
    #remove values outside the region which for these data are set to -9999.
    #pts = pts[pts != -9999.]
    if np.isnan(np.nanmean(pts)):
        #print(point,"nan")
        pts=np.median(data)
        print("returning",pts)

    #return(scipy.stats.nanmean(pts)) #deprecated from scipy 0.15
    return(np.nanmean(pts))


# # Part 1 
# ### Wrangling the raw data
# 
# ### Deposit locations - mine and mineral occurances
# The most significant dataset for this workflow is the currently known locations of mineral occurences. From the data we know about these known deposits we will build a model to predict where future occurences will be.

# In[3]:


#Set the filename
mineshape="SA-DATA/MinesMinerals/mines_and_mineral_occurrences_all.shp"

#read in the file
shapeRead = shapefile.Reader(mineshape)

#And save out some of the shape file attributes
recs    = shapeRead.records()
shapes  = shapeRead.shapes()
fields  = shapeRead.fields
Nshp    = len(shapes)


# In[4]:


mines=[]
for index,f in enumerate(recs[:]):
    mines.append([index,shapes[index].points[0][0],shapes[index].points[0][1],f[3]])


# In[5]:


df = pd.DataFrame(mines, columns =['index','lon','lat','com'])
df=df.set_index('index')


# In[6]:


df


# In[7]:


#Get the gawler map boundary
mineshape="SA-DATA/GCAS_Boundary/GCAS_Boundary.shp"

#read in the file
shapeRead = shapefile.Reader(mineshape)
#And save out some of the shape file attributes
shapes  = shapeRead.shapes()
xval = [x[0] for x in shapes[0].points]
yval = [x[1] for x in shapes[0].points]


# In[8]:


#Plot a few of the chosen target commodities on the map


# # Wrangle the geophysical and geological datasets
# Each geophysical dataset could offer instight into various commodities. Here we load in the pre-formatted datasets and prepare them for further manipulations. 

# ## Resistivity

# In[9]:


data_res=pd.read_csv("SA-DATA/Resistivity/AusLAMP_MT_Gawler.xyzr",
                     sep='\s+',header=0,names=['lat','lon','depth','resistivity'])
data_res.head()


# In[10]:


lon_res=data_res.lon.values
lat_res=data_res.lat.values
depth_res=data_res.depth.values
res_res=data_res.resistivity.values


# In[11]:


f=[]
for i in data_res.depth.unique():
    f.append(data_res[data_res.depth==i].values)

f=np.array(f)
print("Resitivity in:", np.shape(f))


# In[12]:


#Set an array we can interrogate values of later
#This is the same for all resistivity vectors
lonlatres=np.c_[f[0,:,1],f[0,:,0]]
lonres=f[0,:,1]
latres=f[0,:,0]


# ## Faults and dykes

# In[13]:


#Get fault data neo
faultshape="SA-DATA/Neoproterozoic - Ordovician faults_shp/Neoproterozoic - Ordovician faults.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsNeo=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsNeo.append([j[0],j[1]])
faultsNeo=np.array(faultsNeo)


# In[14]:


#Get fault data archean
faultshape="SA-DATA/Archaean - Early Mesoproterozoic faults_shp/Archaean - Early Mesoproterozoic faults.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsArch=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsArch.append([j[0],j[1]])
faultsArch=np.array(faultsArch)


# In[15]:


#Get fault data dolerite dykes swarms
faultshape="SA-DATA/Gairdner Dolerite_shp/Gairdner Dolerite.shp"
shapeRead = shapefile.Reader(faultshape)
shapes  = shapeRead.shapes()
Nshp    = len(shapes)

faultsGair=[]
for i in range(0,Nshp):
    for j in shapes[i].points:
        faultsGair.append([j[0],j[1]])
faultsGair=np.array(faultsGair)


# ### Netcdf formatted

# In[16]:


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


# In[17]:


x1,y1,z1 = readnc("SA-DATA/aster-AlOH-cont.nc")
x2,y2,z2 = readnc("SA-DATA/aster-AlOH-comp.nc")
x3,y3,z3 = readnc("SA-DATA/aster-FeOH-cont.nc")
x4,y4,z4 = readnc("SA-DATA/aster-Ferric-cont.nc")
x5,y5,z5 = readnc("SA-DATA/aster-Ferrous-cont.nc")
x6,y6,z6 = readnc("SA-DATA/aster-Ferrous-index.nc")
x7,y7,z7 = readnc("SA-DATA/aster-MgOH-comp.nc")
x8,y8,z8 = readnc("SA-DATA/aster-MgOH-cont.nc")
x9,y9,z9 = readnc("SA-DATA/aster-green.nc")
x10,y10,z10 = readnc("SA-DATA/aster-kaolin.nc")
x11,y11,z11 = readnc("SA-DATA/aster-opaque.nc")
x12,y12,z12 = readnc("SA-DATA/aster-quartz.nc")
x13,y13,z13 = readnc("SA-DATA/aster-regolith-b3.nc")
x14,y14,z14 = readnc("SA-DATA/aster-regolith-b4.nc")
x15,y15,z15 = readnc("SA-DATA/aster-silica.nc")
x16,y16,z16 = readnc("SA-DATA/sa-base-elev.nc")
x17,y17,z17 = readnc("SA-DATA/sa-dem.nc")
x18,y18,z18 = readnc("SA-DATA/sa-base-dtb.nc")
x19,y19,z19 = readnc("SA-DATA/sa-mag-2vd.nc")
x20,y20,z20 = readnc("SA-DATA/sa-mag-rtp.nc")
x21,y21,z21 = readnc("SA-DATA/sa-mag-tmi.nc")
x22,y22,z22 = readnc("SA-DATA/sa-rad-dose.nc")
x23,y23,z23 = readnc("SA-DATA/sa-rad-k.nc")
x24,y24,z24 = readnc("SA-DATA/sa-rad-th.nc")
x25,y25,z25 = readnc("SA-DATA/sa-rad-u.nc")
x26,y26,z26 = readnc("SA-DATA/sa-grav.nc")


# In[19]:


#Categorised geology
x27,y27,z27 = readnc("SA-DATA/sa-geo-archean.nc")
x28,y28,z28 = readnc("SA-DATA/sa-geol1.nc")


# # Part 2 - Spatial data mining of datasets
# 
# ### Select the commodity and geophysical features to use 
# (edit *commname* and turn labels on/off as required)

# In[52]:


# Set the commoditiy we will explore
commname='Ni'
comm=df[df['com'].str.contains(commname)]
comm=comm.reset_index(drop=True)
print(comm.shape)


# In[53]:


#[print(i) for i in comm.com]


# In[54]:


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


# ### Run the spatial data mining

# In[55]:


#We may want to train and test just over the regions that the grids are valid.
#So we can crop the known deposits to the extent of the grids.
#comm=comm[(comm.lon<max(xval)) & (comm.lon>min(xval)) & (comm.lat>min(yval)) & (comm.lat<max(yval))]

sizes=np.shape(comm)

#Now make a set of "non-deposits" using a random location within our exploration area
lats_rand=np.random.uniform(low=min(df.lat), high=max(df.lat), size=len(comm.lat))
lons_rand=np.random.uniform(low=min(df.lon), high=max(df.lon), size=len(comm.lon))


# In[56]:




#Define a function to coregister the grids. 
#Requires list of lat and lon, will return the value at that point for all the hardcoded grids
def coregLoop(sampleData):
    
    lat=sampleData[0]
    lon=sampleData[1]
    region=1 #degrees
    region2=200

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

    #Categorical data indexes
    xloc27=(np.abs(np.array(x27) - lon).argmin())
    yloc27=(np.abs(np.array(y27) - lat).argmin())
    xloc28=(np.abs(np.array(x28) - lon).argmin())
    yloc28=(np.abs(np.array(y28) - lat).argmin())
    
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
                    z27[yloc27,xloc27],z28[yloc28,xloc28]
                   ])
    coregMap=np.append(vals,[random.randint(-1, 1)])
    
    return(coregMap)
    


# ## Run spatial mining of known deposits and "non-deposits"
# Must be re-run on each commodity change.

# In[ ]:

print("running deps1")
#Interrogate the data associated with deposits
deps1=[]
for row in comm.itertuples():
    lazy_result = coregLoop([row.lat,row.lon])
    deps1.append(lazy_result)


# In[27]:


vec1=pd.DataFrame(np.squeeze(deps1),columns=lons+numerical_features+categorical_features)
vec1['deposit'] = 1 #Add the "depoist category flag"


# In[28]:

print("running deps0")
#Interrogate the data associated with randomly smapled points to use as counter-examples
deps0=[]
for lat,lon in zip(lats_rand,lons_rand):
    lazy_result = coregLoop([lat,lon])
    deps0.append(lazy_result)


# In[29]:


vec2=pd.DataFrame(np.squeeze(deps0),columns=lons+numerical_features+categorical_features)
vec2['deposit'] = 0 #Add the "non-deposit category flag"


# In[30]:


#Remove lots of the zeros
training_data = pd.concat([vec1, vec2], ignore_index=True)


# In[31]:


training_data['sum']=(training_data == 0).astype(int).sum(axis=1)
(training_data == 0).astype(int).sum(axis=1).value_counts()
#If many of the points have no data, drop them
indexNames = training_data[ training_data['sum'] > 10 ].index
training_data.drop(indexNames, inplace=True)
training_data.drop(columns=['sum'], inplace=True)
#indexNames = training_data[ training_data['17dem'] == 0 ].index
#training_data.drop(indexNames , inplace=True)

#Save number of deps/non-deps
lennon=len(training_data.deposit[training_data.deposit==0])
lendep=len(training_data.deposit[training_data.deposit==1])

training_data


# In[32]:


#Save the training data out to a file
training_data.to_csv("training_data-"+commname+".csv")


# ## Run spatial mining of gridded data
# Only needs to be done once. Then the values of the grid are used to predict whatever commodity is run.

# In[33]:


#Make a regularly spaced grid here for use in making a probablilty map later
lats_reg=np.linspace(min(yval),max(yval),100)
lons_reg=np.linspace(min(xval),max(xval),100)

sampleData=[]
for lat in lats_reg:
    for lon in lons_reg:
            sampleData.append([lat, lon])


# In[34]:

print("running deps grid")
gridgawler=[]
tic=time.time()
for geophysparams in sampleData:
    #lazy_result = delayed(coregLoop)(geophysparams)
    lazy_result = coregLoop(geophysparams)
    gridgawler.append(lazy_result)
print("appended, now running...")

#c=compute(gridgawler)
toc=time.time()

print("Time taken:", toc-tic, " seconds")


# In[35]:


target_data=pd.DataFrame(np.squeeze(gridgawler),columns=lons+numerical_features+categorical_features)


# In[36]:


target_data.columns


# In[37]:


#Plot to check consistency of sampled "region" choice and the underlying raw grid.


# # Part 3 - Machine learning model
# 
# What mines/minerals are associated with which solid geology rock unit? 
# 
# Is there a preference, e.g. is GOLD found in ARCHEAN rocks perhaps?
# 
# What whole-rock/layer geophysics/surface is predicitve of which mineral?

# In[188]:


# from sklearn.feature_selection import SelectFromModel
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import train_test_split


# In[47]:


#Create the 'feature vector' and a 'target classification vector'
features=training_data[numerical_features+categorical_features]
targets=training_data.deposit

#Create the ML classifier with numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])


# In[48]:


#Train the random forest
print('Random Forest...')

rf.fit(features,targets)
print("Done RF")

scores = cross_val_score(rf, features,targets, cv=10)
print("RF 10-fold cross validation Scores:", scores)
print("SCORE Mean: %.2f" % np.mean(scores), "STD: %.2f" % np.std(scores), "\n")

plt.plot(targets,'b-',label='Target (expected)')
plt.plot(rf.predict(features),'rx',label='Prediction')
plt.xlabel("Feature set")
plt.ylabel("Target/Prediction")
plt.legend()

#print(rf.predict(features))


# In[50]:


#Print out the feature scores
#rf['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features)
#np.join(numerical_features,rf['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features))
#plt.plot(rf.named_steps['classifier'].feature_importances_)
#print(np.mean(rf.steps[1][1].feature_importances_), np.median(rf.steps[1][1].feature_importances_))

#Just print the significant features above some threshold
for i,lab in enumerate(np.append(numerical_features,rf['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features))):
    if rf.steps[1][1].feature_importances_[i] > 1*np.mean(rf.steps[1][1].feature_importances_): 
        print(i, rf.steps[1][1].feature_importances_[i],lab )


# In[51]:



# In[42]:


#Chec the probabilities at each of the deposit/non-deposit points
print('RF...')
pRF=np.array(rf.predict_proba(features))
print("Done RF")


# ## Finally, apply the model to the grid

# In[43]:


#Apply the trained ML to our gridded data to determine the probabilities at each of the points
print('RF...')
pRF_map=np.array(rf.predict_proba(target_data[numerical_features+categorical_features]))
print("Done RF")


# In[44]:


#Make a function that can turn point arrays into a full meshgrid
def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


# In[45]:


#Create a meshgrid from our xyz list of points
gridX,gridY,gridZ=grid(target_data.lon, target_data.lat,pRF_map[:,1])


# In[46]:


#Plot the final target map
fig = plt.figure(figsize=(10,10),dpi=150)

#Make a map projection to plot on.
ax = plt.axes(projection=ccrs.PlateCarree())

#Put down a base map
ax.stock_img()
ax.coastlines(resolution='10m', color='gray',)
tiler = Stamen('terrain-background')
mercator = tiler.crs
ax.add_image(tiler, 6)

#Make the gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.3, color='gray', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.bottom_labels = True
gl.right_labels = False
gl.left_labels = True
#gl.xlines = False
gl.xlocator = mticker.FixedLocator(list(np.linspace(np.floor(min(df.lon))+1,np.ceil(max(df.lon))-1,num=5)))
gl.ylocator = mticker.FixedLocator(list(np.linspace(np.floor(min(df.lat))+1,np.ceil(max(df.lat))-1,num=5)))
gl.xlocator = mticker.FixedLocator([141,138,135,132,129])
gl.ylocator = mticker.FixedLocator([-38,-34,-31,-28,-26])
#gl.ylocator = mticker.FixedLocator(list(np.linspace(-28,-35,num=3)))
gl.xlabel_style = {'size': 10, 'color': 'gray'}
gl.ylabel_style = {'size': 10, 'color': 'gray'}
#gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

img_extent = [min(df.lon)+1,  max(df.lon)-1, min(df.lat)+1,max(df.lat)-1]
ax.set_extent(img_extent, ccrs.PlateCarree())
#ax.margins(0.05) #adds 5% padding to the autoscaling

#Create a patch of the gawler region where the data is
path=Path(list(zip(xval, yval)))
patch = PathPatch(path, facecolor='none')
plt.gca().add_patch(patch)

#Plot the main map
#im=ax.contourf(gridX,gridY,gridZ,cmap=plt.cm.coolwarm)
im = ax.imshow(gridZ, interpolation='bicubic', cmap=plt.cm.bwr,
                origin='lower', extent=[np.min(gridX),np.max(gridX),np.min(gridY),np.max(gridY)],
                clip_path=patch, clip_on=True,zorder=1)

#Add the deposits coloured by their classification score
# l4=ax.scatter(training_data.lon[training_data.deposit==0], training_data.lat[training_data.deposit==0],
#               edgecolor='k',s=20,marker='s',
#               c=pRF[lendep:,1],cmap=plt.cm.bwr,vmin=0,vmax=1,zorder=3,label='Non-deposits')

l3=ax.scatter(training_data.lon[training_data.deposit==1], training_data.lat[training_data.deposit==1], 
              edgecolor='k',s=20,marker='o',
              c=pRF[:lendep,1],cmap=plt.cm.bwr,vmin=0,vmax=1,zorder=3,label='Known '+commname+' deposits')


#Plot the outline of the Gawler region
plt.plot(xval,yval,'k--',label='Gawler',linewidth=0.5)


# Add a map title, legend, colorbar
#plt.title('Known deposits and predictive map for Gawler region, SA')
ax.legend(loc=1)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#Make a Colorbar
cbaxes = fig.add_axes([0.16, 0.25, 0.25, 0.015])
cbar = plt.colorbar(l3, cax = cbaxes,orientation="horizontal")
cbar.set_label(commname+' prediction')

plt.show()


# In[ ]:




