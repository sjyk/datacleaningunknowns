# A Data Quality Metric (DQM): How to Estimate The Number of Undetected Errors in Data Sets
Data cleaning, whether manual or algorithmic, is rarely perfect leaving a dataset with an unknown number of false positives and false negatives after cleaning. In many scenarios, quantifying the number of remaining errors is challenging because our data integrity rules themselves may be incomplete, or the available gold-standard datasets may be too small to extrapolate. As the use of inherently fallible crowds becomes more prevalent in data cleaning problems, it is important to have estimators to quantify the extent of such errors. We propose novel species estimators to estimate the number of distinct remaining errors in a dataset after it has been cleaned by a set of crowd workers -- essentially, quantifying the utility of hiring additional workers to clean the dataset. This problem requires new estimators that are robust to false positives and false negatives, and we empirically show on three real-world datasets that existing species estimators are unstable for this problem, while our proposed techniques quickly converge.

This is outdated repository, whcih contains our VLDB experiments.
