# Know-your-Neighbors-Efficient-Author-Profiling-via-Follower-Tweets


# Abstract
User profiling based on social media data is becoming an increasingly relevant task with applications in advertising, forensics, literary studies and sociolinguistic research. Even though profiling of users based on their textual data is possible, social media such as Twitter offer also insight into the data of a given user’s followers. The purpose of this work was to explore how such follower data can be used for profiling a given user, what are its limitations and whether performances, similar to the ones observed when considering a given user’s data directly can be achieved. In this work we present our approach, capable of extracting various feature types and, via sparse matrix factorization, learn a dense, low-dimensional representations of individual persons solely from their followers’ tweet streams. The proposed approach scored second in the PAN 2020 Celebrity profiling shared task, and is computationally non-demanding.

# Prerequired dependencies

``` 
gensim==3.8.3
numpy==1.18.5
tqdm==4.47.0
pandas==1.0.5
nltk==3.5
scipy==1.5.0
scikit_learn==0.23.2
xgboost==1.2.1
```

# Data

The dataset for this task is provided by the PAN workshop organizators of the CLEF'20 conference. The link to the dataset can be found on the following link:

`` https://zenodo.org/record/3691922#.X4mA1E8zaEI ``


For the training of the models we used 10 randomly selected tweets of 20 random followers of a celebrity. For each model this data is placed in it's corresponding **train_data** folders.

# Model training

We propose following models:

- **ff_big** 
- **ff_celeb**
- **ff_clf_all**
- **ff_clf_avg**
- **ff_larger**
- **ff_splits**
- **ff_splits_all**

Each model has it's own ``ff.sh`` script that should be run. 


# Contribution


### This code was developed by Boshko Koloski & Blaž Škrlj


# Citation

If you use our code please cite our work. 

```
@InProceedings{koloski:2020b,
  author =              {Bo{\v s}ko Koloski and Senja Pollak and Bla{\v z} {\v S}krlj},
  booktitle =           {{CLEF 2020 Labs and Workshops, Notebook Papers}},
  crossref =            {pan:2020},
  editor =              {Linda Cappellato and Carsten Eickhoff and Nicola Ferro and Aur{\'e}lie N{\'e}v{\'e}ol},
  month =               sep,
  publisher =           {CEUR-WS.org},
  title =               {{Know your Neighbors: Efficient Author Profiling via Follower Tweets---Notebook for PAN at CLEF 2020}},
  url =                 {},
  year =                2020
}
```


# Aknowledgements

The work of the last author was funded by the Slovenian Research Agency through a
young researcher grant. The work of other authors was supported by the Slovenian
Research Agency (ARRS) core research programme Knowledge Technologies
(P2-0103), an ARRS funded research project Semantic Data Mining for Linked Open
Data (financed under the ERC Complementary Scheme, N2-0078) and European
Unions Horizon 2020 research and innovation programme under grant agreement No ´
825153, project EMBEDDIA (Cross-Lingual Embeddings for Less-Represented
Languages in European News Media).