## Data Source

Kmer model is from [Oxford Nanopore Technologies](https://github.com/nanoporetech/kmer_models/tree/master/legacy/legacy_r9.4_180mv_450bps_6mer).

Fast5 reads are from CADDE Centre and are downloaded using the script from [Rawhash](https://github.com/CMU-SAFARI/RawHash/tree/main):

```shell
wget -qO-  https://cadde.s3.climb.ac.uk/SP1-raw.tgz | tar -xzv; rm README
```

Move the files in `/dataset/all_id.txt` to a new file, and use [ont_fast5_api](https://github.com/nanoporetech/ont_fast5_api) to generate a single fast5 file:

```shell
multi_to_single_fast5 --input_path R94_COVID --save_path R94_COVID_10k --recursive
```

