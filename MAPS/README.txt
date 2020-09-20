README.txt

This zip directory contains the GeoTIFF formatted grids which can be used as exploration targeting maps.

The GeoTiffs contain 1 data band which contains the "predicted classification probabilities" between 0.0 and 1.0. Values closer to 1.0 have a higher probability of being a mineral deposits, values closer to 0.0 have a lower probability of being a mineral deposit.

Each grid has the same shape and resolution. Each grid represents a different commodity target, or else a different method, distinguished in the description: 

---------------------------------------------
List of files in directory, Grid description
---------------------------------------------
 Targets-Ag.tif, Silver
 Targets-Au-Model2.tif, Gold using preferred Model II for deposit selection method
 Targets-Au.tif, Gold using Model I for deposit selection method
 Targets-Co.tif, Cobalt
 Targets-Cu-Model2.tif, Copper using preferred Model II for deposit selection method
 Targets-Cu-Model2-nores.tif, Copper using preferred Model II for deposit selection method and not usinng AusLAMP resitivity layers.
 Targets-Cu.tif, Copper using Model I selection method
 Targets-DIA.tif, Diamond
 Targets-Fe.tif, Iron
 Targets-Mn.tif, Manganese
 Targets-Ni.tif, Nickel
 Targets-U.tif, Uranium
 Targets-Zn.tif, Zinc
---------------------------------------------

Grids were generated from *.csv data (Longitude, Latitude, Probability), exported from the submitted Python notebook, and cropped to the provided Gawler region, using gdal with the following two commands:

 gdal_translate TARGET.csv -of GTiff TARGET.nc
 gdalwarp TARGET.nc -cutline GCAS_Boundary.shp -t_srs EPSG:7844 -of GTiff TARGET.tif

---------------------------------------------
GeoTIFF grid specifications
---------------------------------------------
Driver: GTiff/GeoTIFF
Files: *.tif
Size is 702, 533
Coordinate System is:
GEOGCS["GDA2020",
    DATUM["Geocentric_Datum_of_Australia_2020",
        SPHEROID["GRS 1980",6378137,298.2572221010042,
            AUTHORITY["EPSG","7019"]],
        AUTHORITY["EPSG","1168"]],
    PRIMEM["Greenwich",0],
    UNIT["degree",0.0174532925199433],
    AUTHORITY["EPSG","7844"]]
Origin = (130.995008542000051,-27.339986651001052)
Pixel Size = (0.009999999999994,-0.009999999999994)
Metadata:
  AREA_OR_POINT=Area
Image Structure Metadata:
  INTERLEAVE=BAND
Corner Coordinates:
Upper Left  ( 130.9950085, -27.3399867) (130d59'42.03"E, 27d20'23.95"S)
Lower Left  ( 130.9950085, -32.6699867) (130d59'42.03"E, 32d40'11.95"S)
Upper Right ( 138.0150085, -27.3399867) (138d 0'54.03"E, 27d20'23.95"S)
Lower Right ( 138.0150085, -32.6699867) (138d 0'54.03"E, 32d40'11.95"S)
Center      ( 134.5050085, -30.0049867) (134d30'18.03"E, 30d 0'17.95"S)
Band 1 Block=702x2 Type=Float32, ColorInterp=Gray