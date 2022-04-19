# Code for Butterworth and Barnett-Moore 2020
Spatial data mining with machine learning to reveal mineral exploration targets under cover in the Gawler Craton, South Australia.
The pre-print manuscript is availble in the "GAWLER" directory.

## Code
Use Mines and Minerals dataset to teach the model values from the datasets that are associated with mines and known deposit locations. Then apply that trained algorithm on the entire area.

The dockerfile will build an environment which has all the required packages.

Use **results-sa.ipynb** for exploring individual commodities

Use **results-sa-test1.py** to generate the first part of the target test set in the Gawler region.

Use **results-sa-test2.py** to do the second part of the target test set generation. 

## Data

Regularized input data is in **SA-DATA** folder.

Model output (exploration targeting maps) are in **MAPS**.

Model training and target datasets generated in **ML-DATA**.

References and links to original data licenses can be found in the manuscript.
