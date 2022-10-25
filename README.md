TODO: update Python version scripts, run `module list` to find name, preferable Python version 3.9+.



```shell
export RUG_NUMBER=<your p-, or s- number>
```

Copying dataset to Peregrine (local):
```shell
tar -czvf dataset.tar.gz dataset/
scp -r "${PWD}"/dataset.tar.gz s3105970@peregrine.hpc.rug.nl:/home/s3105970/training
```

Copying project files to Peregrine (local):
```shell
tar -czvf application.tar.gz --exclude='__pycache__' *.py *.txt *.md preprocessors/ scrapers/ peregrine/ tools/ preprocessed/ 
scp -r "${PWD}"/application.tar.gz s3105970@peregrine.hpc.rug.nl:/home/s3105970/training
```

Extracting project files on Peregrine (peregrine, ~/training):
```shell
```shell
tar -xzvf application.tar.gz
```

Running the Play Store scraper (peregrine, ~/training/peregrine/):
```shell
```shell
sbatch scraper.peregrine.sh
```

Compress job output (peregrine, ~/training/):
```shell
tar -czvf results.tar.gz models results preprocessed
```

Retrieving output from Peregrine (local)
```shell
scp s3105970@peregrine.hpc.rug.nl:/home/s3105970/training/results.tar.gz "${PWD}"/results.tar.gz
```

Run the job on Peregrine, either `sbatch test.peregrine.sh` or `sbatch train.peregrine.sh`.
