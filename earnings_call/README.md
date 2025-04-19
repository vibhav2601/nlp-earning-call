---
license: cc0-1.0
task_categories:
- text-classification
language:
- en
tags:
- finance
pretty_name: Earnings Calls Dataset
size_categories:
- 10K<n<100K
dataset_info:
- config_name: stock_prices
  features:
  - name: date
    dtype: date64
  - name: open
    dtype: float32
  - name: high
    dtype: float32
  - name: low
    dtype: float32
  - name: close
    dtype: float32
  - name: adj_close
    dtype: float32
  - name: volume
    dtype: int64
  - name: company
    dtype: string
  splits:
  - name: train
    num_bytes: 578818
    num_examples: 13155
  download_size: 290243
  dataset_size: 578818
- config_name: transcript-sentiment
  features:
  - name: text
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': negative
          '1': positive
  - name: company
    dtype: string
  - name: date
    dtype: date64
  - name: para_no
    dtype: int32
  splits:
  - name: train
    num_bytes: 7414686
    num_examples: 6851
  - name: test
    num_bytes: 1928515
    num_examples: 1693
  download_size: 3868059
  dataset_size: 9343201
- config_name: transcripts
  features:
  - name: company
    dtype: string
  - name: date
    dtype: date64
  - name: transcript
    dtype: string
  splits:
  - name: train
    num_bytes: 9592380
    num_examples: 150
  - name: test
    num_bytes: 2458569
    num_examples: 38
  download_size: 3577816
  dataset_size: 12050949
---
# Dataset Card for Earnings Calls Dataset

## Dataset Description

- **Homepage:** https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/TJE0D0
- **Paper:** https://www.preprints.org/manuscript/202102.0424/v1
- **Point of Contact:** [Francesco Lelli](https://francescolelli.info/)

### Dataset Summary

The dataset reports a collection of earnings call transcripts, the related stock prices, and the sector index In terms of volume,
there is a total of 188 transcripts, 11970 stock prices, and 1196 sector index values. Furthermore, all of these data originated 
in the period 2016-2020 and are related to the NASDAQ stock market. Furthermore, the data collection was made possible by Yahoo 
Finance and Thomson Reuters Eikon. Specifically, Yahoo Finance enabled the search for stock values and Thomson Reuters Eikon 
provided the earnings call transcripts. Lastly, the dataset can be used as a benchmark for the evaluation of several NLP techniques
to understand their potential for financial applications. Moreover, it is also possible to expand the dataset by extending the period
in which the data originated following a similar procedure.





### Citation Information

```bibtex
@data{TJE0D0_2021,
author = {Roozen, Dexter and Lelli, Francesco},
publisher = {DataverseNL},
title = {{Stock Values and Earnings Call Transcripts: a Sentiment Analysis Dataset}},
year = {2021},
version = {V1},
doi = {10.34894/TJE0D0},
url = {https://doi.org/10.34894/TJE0D0}
}
```

