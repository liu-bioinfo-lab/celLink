# CelLink: Integrate single-cell multi-omics data with few linked features and imbalanced cell populations
<img src="docs/images/CelLink_logo.png" width="200">

## About CelLink
CelLink is a Python package designed for single-cell multi-omics integration. It excels uniquely in integrating datasets with weak feature linkage and imbalanced cell populations. CelLink normalizes and smooths feature profiles to align scales across datasets and integrates them through a multi-phase pipeline that iteratively employs the optimal transport algorithm. It dynamically refines cell-cell correspondences, identifying and excluding cells that cannot be reliably matched, thus avoiding performance degradation caused by erroneous imputations. A classic example of weak linkage is seen in the integration of scRNA-seq and CODEX (spatial proteomic data) from the Human Pancreas Analysis Program (HPAP). 

<img src="docs/images/pipeline.png" width="1000">
