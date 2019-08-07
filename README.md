# GPU accelerated image registration

## How to run the program
First, create the python environment if you do not have one set up already.
```
$ cd imagereg
$ conda env create -f environment.yml
$ conda activate imagereg
```


Then there are two options for running the program: from the command line, and from within python.

### 1. From the command line
```
conda activate imagereg
mkdir output_directory
python path\to\imagereg\main.py tests\images img[0-9].tif output_directory
```

### 2. From within python

```
$ conda activate imagereg
$ python
>>> import imagereg
>>> imagereg.pipeline('tests\images', 'img[0-9].tif', 'output_directory')
```

## Setting up your development environment

1. Create a conda environment
```
conda create -n imagereg -c conda-forge python=3.6.8 pip click numpy=1.16.4 pandas=0.24.2 scikit-image=0.15.0 pytest
source activate imagereg
```
2. Optional, for GPU acceleration: Install CUDA version 10.0
3. Optional, for GPU acceleration: pip install cupy version 5.4.0 into the imagereg conda environment
4. Install imagereg with pip into your environment:
```
pip install .
```
