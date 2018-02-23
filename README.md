coremagmodels
=====
This package provides Python code to calculate magnetic fields for a variety of global core magnetic field models. As of now, these include:
* [CHAOS-6](http://www.spacecenter.dk/files/magnetic-models/CHAOS-6/)
* [GUFM1](http://jupiter.ethz.ch/~cfinlay/gufm1.html) 
* [GUFM-sat-E3](http://jupiter.ethz.ch/~cfinlay/gufm1.html) 
* [GUFM-sat-Q2](http://jupiter.ethz.ch/~cfinlay/gufm1.html) 
* [GUFM-sat-Q3](http://jupiter.ethz.ch/~cfinlay/gufm1.html) 
future implementations might add
* [COV-OBS.x1](http://www.spacecenter.dk/files/magnetic-models/COV-OBSx1/)
* IGRF
* WMM

Usage
-----
To use, simply clone the git directory, then either run the code from the directory directly, or run `python setup.py install` to allow for importing the method in any python project. A few different routines are currently available, allowing users to calculate radial field strength at any spatial location, rms field strength over the whole core, and other metrics of core power spectrum.

Users should be able to be calculate parameters for any year from 1590 to 1990 from the main gufm1_data.txt file, but you can also use data files from individual years downloaded directly from his website if desired.

For examples, see the [GUFM1 Demonstration](examples/GUFM1 Demonstration.ipynb) ipython notebook.


Citations
--------
####GUFM1 -  [website](http://jupiter.ethz.ch/~cfinlay/gufm1.html)  
> Jackson, A., Jonkers, A. R., & Walker, M. R. (2000). Four centuries of geomagnetic secular variation from historical records. Philosophical Transactions of the Royal Society of London A: Mathematical, Physical and Engineering Sciences, 358(1768), 957-990.

####CHAOS-6 - [website](http://www.spacecenter.dk/files/magnetic-models/CHAOS-6/) 
> Finlay, C. C., Olsen, N., Kotsiaros, S., Gillet, N., & Clausen, L. T. (2016). Recent geomagnetic secular variation from Swarm and ground observatories as estimated in the CHAOS ‑ 6 geomagnetic field model. *Earth, Planets and Space*, 1–18. https://doi.org/10.1186/s40623-016-0486-1

