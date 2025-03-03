> ⚠️ **Note:** 
> Pressure makes diamonds, but it doesn’t always make perfectly structured code.
> A full refactoring is still on the to-do list—please note that this project doesn’t reflect my views on coding standards.

# rna_sequencing

Single Cell RNA Sequencing Project @ Machine Learning for Genomic Data Science @ LUH

## Programming Environment

### Create the Datasets

```shell
cd data
python create_splits.py
```

### Python Version

We use python 3.10.

### Virtual Environment

It is advisable to use [venv](https://docs.python.org/3/library/venv.html).

Install requirements with:

```shell
$ pip install -r requirements.txt
```

And save requirements with:

```shell
pip freeze > requirements.txt
```

## Folder Structure

-   [utils](./utils) - contains general code usable in different experiments

-   [experiments](./experiments) - contains experiments carried out

-   [data](./data) - contains different datasets

-   [data/original](./data/original) original dataset provided by course

-   [data/splits](./data/splits) split dataset for normation

We use Readme.md extensively to document experiments and other content.
