# term-tagger

term-tagger is a tool for bacteria transcription terminator prediction based on BERT-CRF model.  

## dependency
- required (for terminator prediction/model inference): 
  - pytorch: test on version `1.7.0+cu110`, other version should work
  - transformers: test on version `4.18.0`
  - pyfaidx: test on version `0.7.1`
- optional :
   - ushuffle (for model training)
   - biopython (for model training) : test on version `1.79`
   - bedtools (for evaluate genomic distribution of predicted terminator)


## usage

- `term-tagger` takes bacteria genome sequence (can be contig or complete/draft genome) as input, and produces predicted terminator coordinate and strand information with confidence scores in bed format







