# ee364b-transfer-learning

## Data
Data is obtained from: Won, K., Kwon, M., Ahn, M. et al. EEG Dataset for RSVP and P300 Speller Brain-Computer Interfaces. Sci Data 9, 388 (2022). https://doi.org/10.1038/s41597-022-01509-w. 

## Preprocessing
Some of the preprocessing steps are taken directly from the original paper with the data: https://github.com/Kyungho-Won/EEG-dataset-for-RSVP-P300-speller. 

## Algorithms to Calculate Centers
Euclidean mean is from: P. Zanini, M. Congedo, C. Jutten, S. Said and Y. Berthoumieu, "Transfer Learning: A Riemannian Geometry Framework With Applications to Brain–Computer Interfaces," in IEEE Transactions on Biomedical Engineering, vol. 65, no. 5, pp. 1107-1116, May 2018, doi: 10.1109/TBME.2017.2742541.

Riemannian mean is from: Barbaresco, F. (2009). Interactions between Symmetric Cone and Information Geometries: Bruhat-Tits and Siegel Spaces Models for High Resolution Autoregressive Doppler Imagery. In: Nielsen, F. (eds) Emerging Trends in Visual Computing. ETVC 2008. Lecture Notes in Computer Science, vol 5416. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-00826-9_6.
Note that the algorithm is closely adapted from: Alexandre Barachant, Quentin Barthélemy, Jean-Rémi King, Alexandre Gramfort, Sylvain Chevallier, Pedro L. C. Rodrigues, Emanuele Olivetti, Vladislav Goncharenko, Gabriel Wagner vom Berg, Ghiles Reguig, Arthur Lebeurrier, Erik Bjäreholt, Maria Sayu Yamamoto, Pierre Clisson, and Marie-Constance Corsi. pyriemann/pyriemann: v0.5, June 2023. https://doi.org/10.5281/zenodo.8059038

Matrix median is from: S. Setzer, G. Steidl, T. Teuber. On vector and matrix median computation, Journal of Computational and Applied Mathematics, Volume 236, Issue 8, 2012, Pages 2200-2222, ISSN 0377-0427, https://doi.org/10.1016/j.cam.2011.09.042.

Riemannian median is from: P. Thomas Fletcher, Suresh Venkatasubramanian, Sarang Joshi, The geometric median on Riemannian manifolds with application to robust atlas estimation, NeuroImage, Volume 45, Issue 1, Supplement 1, 2009, Pages S143-S152, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2008.10.052.
Note that the algorithm is closely adapted from: Alexandre Barachant, Quentin Barthélemy, Jean-Rémi King, Alexandre Gramfort, Sylvain Chevallier, Pedro L. C. Rodrigues, Emanuele Olivetti, Vladislav Goncharenko, Gabriel Wagner vom Berg, Ghiles Reguig, Arthur Lebeurrier, Erik Bjäreholt, Maria Sayu Yamamoto, Pierre Clisson, and Marie-Constance Corsi. pyriemann/pyriemann: v0.5, June 2023. https://doi.org/10.5281/zenodo.8059038

Huber Centroid is from: I. Ilea, L. Bombrun, R. Terebes, M. Borda and C. Germain, "An M-Estimator for Robust Centroid Estimation on the Manifold of Covariance Matrices," in IEEE Signal Processing Letters, vol. 23, no. 9, pp. 1255-1259, Sept. 2016, doi: 10.1109/LSP.2016.2594149.

## Plots
Some of the plots are taken from other sources:

EEG.png: Won, K., Kwon, M., Ahn, M. et al. EEG Dataset for RSVP and P300 Speller Brain-Computer Interfaces. Sci Data 9, 388 (2022). https://doi.org/10.1038/s41597-022-01509-w. 

p300-speller.jpg: Krusienski DJ, Sellers EW, McFarland DJ, Vaughan TM, Wolpaw JR. Toward enhanced P300 speller performance. J Neurosci Methods. 2008 Jan 15;167(1):15-21. doi: 10.1016/j.jneumeth.2007.07.017. Epub 2007 Aug 1. PMID: 17822777; PMCID: PMC2349091.