# CelLink: integrating single-cell multi-omics data with weak feature linkage and imbalanced cell populations
[![PyPI](https://img.shields.io/pypi/v/scanpy?logo=PyPI)](https://pypi.org/project/cellink-luo/)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Review](https://views.whatilearened.today/views/github/liu-bioinfo-lab/CelLink.svg)
[![Docs](https://readthedocs.com/projects/icb-scanpy/badge/?version=latest)](https://cellink.readthedocs.io/en/latest/)

## About CelLink
CelLink is a Python package designed for single-cell multi-omics integration. It excels uniquely in integrating datasets with weak feature linkage and imbalanced cell populations. CelLink normalizes and smooths feature profiles to align scales across datasets and integrates them through a multi-phase pipeline that iteratively employs the optimal transport algorithm. It dynamically refines cell-cell correspondences, identifying and excluding cells that cannot be reliably matched, thus avoiding performance degradation caused by erroneous imputations. A classic example of weak linkage is seen in the integration of scRNA-seq and CODEX (spatial proteomic data) from the Human Pancreas Analysis Program (HPAP). 

<img src="docs/images/pipeline.png" width="700">

## Novel capabilities of CelLink 
CelLink uniquely enables cell subtype annotation, correction of mislabelled cells, and spatial transcriptomic analyses by imputing transcriptomic profiles for spatial proteomics data. Its great ability to impute large-scale paired single-cell multi-omics profiles positions it as a pivotal tool for building single-cell multi-modal foundation models.


## Installation
Cellink can be installed from PyPI using pip. For best practices, create a new virtual environment before installation. Below, we demonstrate how to set up this environment using conda.

```bash
conda create -n CelLink python=3.9
conda activate CelLink
pip install cellink-luo
```

## Vignettes
A tutorial on integrating scRNA-seq and CODEX datasets from donor HPAP023 is provided. The feature linkage information between coding genes and proteins is stored in [protein_gene_relationship.csv](docs/protein_gene_relationship.csv). Please check our [tutorial website](https://cellink.readthedocs.io/en/latest/index.html).

## Citation
If you use Cellink in your research, please kindly cite our paper using the following reference:
[Xin Luo et al. CelLink: Integrate single-cell multi-omics data with few linked features and imbalanced cell populations, Biorxiv 2024](https://doi.org/10.1101/2024.11.08.622745).

## License
This project is licensed under the GNU General Public License v3.0. For more information, see the [LICENSE](LICENSE) file in this repository.
