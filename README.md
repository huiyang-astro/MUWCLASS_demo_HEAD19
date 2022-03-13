# MUWCLASS_demo
 
## MUWCLASS Pipeline on CSCv2 Demonstration
### Hui Yang1, Jeremy Hare2, Oleg Kargaltsev1, Igor Volkov1
### 1 The George Washington University 2 NASA GSFC

### contact huiyang@gwu.edu if you have any questions

This notebook presents a demonstration of classifying Chandra Source Catalog v2 (CSCv2) using the MUltiWavelength Machine Learning CLASSification Pipeline with CSCv2 and multiwavelength data

* This notebook was run in CIAO 4.14 with Python 3.9 
* conda create -n ciao-4.14 -c https://cxc.cfa.harvard.edu/conda/ciao -c conda-forge ciao sherpa ds9 ciao-contrib caldb_main marx python=3.9

* run 'bash install-packages.sh' under CIAO 4.14 environment to install all required packages 

* then, make sure to enable widgetsnbextension and ipyaladin, run 
* jupyter nbextension enable --py widgetsnbextension
* jupyter nbextension enable --py --sys-prefix ipyaladin
on your terminal 

* You might also need to manually register the existing ds9 with the xapns name server by selecting the ds9 File->XPA->Connect menu option so your ds9 will be fully accessible to pyds9. 

