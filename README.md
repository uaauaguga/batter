# batter

- batter is a tool for bacteria transcription terminator prediction based on BERT-CRF model. 
- It aims to predict both rho dependent and rho independent terminator across diverse species in Bacteria domain.

## Installation

### dependency

- for inference, the following python packages are required
  - [pytorch](https://pytorch.org/): test on version `1.7.0+cu110`, other version should work
  - [transformers](https://huggingface.co/docs/transformers/index): test on version `4.18.0`
  - [pyfaidx](https://pythonhosted.org/pyfaidx/): test on version `0.7.1`
  - [numpy](https://numpy.org/)

- optional :
  - [bedtools](https://bedtools.readthedocs.io/) (for evaluate genomic distribution of predicted terminator)

### Installation


```{bash}
git clone https://github.com/uaauaguga/batter.git 
```


## usage

- `batter` takes bacteria genome sequence (can be contig or complete/draft genome) as input, and produces predicted terminator coordinate and strand information with confidence scores in bed format

```bash
#batter scan top strand by default
scripts/batter --fasta examples/example.fa  --output examples/example.bed --device cuda:0

# if you want to scan both up strand and bottom strand, use -r option
scripts/batter --fasta examples/example.fa  --output examples/example.bed --device cuda:0 -rc
```

