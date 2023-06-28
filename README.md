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

## Inference

```bash
#batter scan top strand by default
scripts/batter --fasta examples/example.fa  --output examples/example.bed --device cuda:0

# if you want to scan both up strand and bottom strand, use --reverse-complement/-rc option
scripts/batter --fasta examples/example.fa  --output examples/example.bed --device cuda:0 -rc

# if you want to keep temperorary file, use --keep-temp/-kt option
# you can also specify path of temporary file with parameter "--tmp-file"
scripts/batter --fasta examples/example.fa  --output examples/example.bed --device cuda:0 -rc -kt


# if more efficient scanning (at cost of lower sensitivity) is desired, you can increase the step size (100 nt by default) for scanning 
scripts/batter --fasta examples/example.fa  --output examples/example.250.rc.bed --device cuda:0 --stride 250 -rc

```

## Model calibration

 The output of batter are genomic intervals associated with a score. The predictions are quite conservative, but for low GC-content bacteria species, the false positive rate tend to be relatively higher. There are several choices to determine which cutoff to use. 

- A cutoff of 0.5 is generally safe.
- The mean cutoffs of species with different genomic GC content that achieve a false positive rate (FPR) of 0.1/KB are listed as follow:

| GC content | cutoff |
| ---------- | ------ |  
| 20%-30%    | 0.547  |
| 30%-40%    | 0.533  |
| 40%-50%    | 0.511  |
| 50%-60%    | 0.496  |
| 60%-70%    | 0.491  |
| 70%-80%    | 0.489  |

- Provide your genome, calculate tetra-mer frequency (TNF), and predict cutoff at FPR of 0.1 with a pretrained model (this feature depends on lightgbm package) 

```bash
scripts/tnf-score-cutoff-predictor.py -tnf examples/example.TNF --scores examples/example.cutoff.01
```

- Of course you can simulate a background sequence set, and determine the cutoff yourself.


## Annotation

- Computational annotation of prokaryote genome, especially annotation of proteining coding genes, is relatively reliable, and you can annotate predicted terminators with its relative position to protein coding gene
- You can download bacteria genome annotation from ncbi, or perform annotation using tools like [prokka](https://github.com/tseemann/prokka). You'll get a file in gff format.
- 

