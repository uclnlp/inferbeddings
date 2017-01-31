# Inferbeddings [![wercker status](https://app.wercker.com/status/47cfd21ce8315fb361ac5b4247ce280a/s/master "wercker status")](https://app.wercker.com/project/byKey/47cfd21ce8315fb361ac5b4247ce280a)

Rule Injection in Knowledge Graph Embeddings via Adversarial Training.


Usage:

```
$ python3 setup.py install --user
running install
running bdist_egg
running egg_info
[...]
$ ./bin/adv-cli.py -h
usage: Rule Injection via Adversarial Training [-h] --train TRAIN [--valid VALID] [--test TEST] [--lr LR] [--nb-batches NB_BATCHES] [--nb-epochs NB_EPOCHS] [--model MODEL] [--similarity SIMILARITY]
                                               [--objective OBJECTIVE] [--margin MARGIN] [--embedding-size EMBEDDING_SIZE] [--predicate-embedding-size PREDICATE_EMBEDDING_SIZE] [--auc] [--seed SEED]
                                               [--clauses CLAUSES] [--adv-lr ADV_LR] [--adv-nb-epochs ADV_NB_EPOCHS] [--adv-weight ADV_WEIGHT] [--adv-margin ADV_MARGIN] [--adv-restart] [--save SAVE]

optional arguments:
  -h, --help                                                                                  show this help message and exit
  --train TRAIN, -t TRAIN
  --valid VALID, -v VALID
  --test TEST, -T TEST
  --lr LR, -l LR
  --nb-batches NB_BATCHES, -b NB_BATCHES
  --nb-epochs NB_EPOCHS, -e NB_EPOCHS
  --model MODEL, -m MODEL                                                                     Model
  --similarity SIMILARITY, -s SIMILARITY                                                      Similarity function
  --objective OBJECTIVE, -o OBJECTIVE                                                         Loss function
  --margin MARGIN, -M MARGIN                                                                  Margin
  --embedding-size EMBEDDING_SIZE, --entity-embedding-size EMBEDDING_SIZE, -k EMBEDDING_SIZE  Entity embedding size
  --predicate-embedding-size PREDICATE_EMBEDDING_SIZE, -p PREDICATE_EMBEDDING_SIZE            Predicate embedding size
  --auc, -a                                                                                   Measure the predictive accuracy using AUC-PR and AUC-ROC
  --seed SEED, -S SEED                                                                        Seed for the PRNG
  --clauses CLAUSES, -c CLAUSES                                                               File containing background knowledge expressed as Horn clauses
  --adv-lr ADV_LR, -L ADV_LR                                                                  Adversary learning rate
  --adv-nb-epochs ADV_NB_EPOCHS, -E ADV_NB_EPOCHS                                             Adversary number of training epochs
  --adv-weight ADV_WEIGHT, -W ADV_WEIGHT                                                      Adversary weight
  --adv-margin ADV_MARGIN                                                                     Adversary margin
  --adv-restart, -R                                                                           Restart the optimization process for identifying the violators
  --save SAVE                                                                                 Path for saving the serialized model

```

If the parameter `--adv-lr` is not specified, the method does not perform any adversarial training -- i.e. it simply trains the Knowledge Graph Embedding models by minimizing a standard pairwise loss in, such as the margin-based ranking loss in [1].

Example - Embedding the WN18 Knowledge Graph:

```
$ ./bin/adv-cli.py --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model ComplEx --similarity dot --margin 5 --embedding-size 100 --nb-epochs 100
INFO:adv-cli.py:Command line: --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model ComplEx --similarity dot --margin 5 --embedding-size 100 --nb-epochs 100
INFO:adv-cli.py:#Training Triples: 141442, #Validation Triples: 5000, #Test Triples: 5000
INFO:adv-cli.py:#Entities: 40943        #Predicates: 18
INFO:adv-cli.py:Samples: 141442, no. batches: 10 -> batch size: 14145
INFO:adv-cli.py:Epoch: 1/1      Loss: 7.7348 ± 1.9103
INFO:adv-cli.py:Epoch: 1/1      Fact Loss: 1094047.0703
INFO:adv-cli.py:Epoch: 2/1      Loss: 1.6843 ± 0.4089
INFO:adv-cli.py:Epoch: 2/1      Fact Loss: 238233.4922
INFO:adv-cli.py:Epoch: 3/1      Loss: 0.5919 ± 0.1695
INFO:adv-cli.py:Epoch: 3/1      Fact Loss: 83726.5850
INFO:adv-cli.py:Epoch: 4/1      Loss: 0.2362 ± 0.0336
INFO:adv-cli.py:Epoch: 4/1      Fact Loss: 33412.5703
INFO:adv-cli.py:Epoch: 5/1      Loss: 0.1120 ± 0.0215
INFO:adv-cli.py:Epoch: 5/1      Fact Loss: 15836.4889
[..]
INFO:adv-cli.py:Epoch: 97/1     Loss: 0.0026 ± 0.0008
INFO:adv-cli.py:Epoch: 97/1     Fact Loss: 371.5449
INFO:adv-cli.py:Epoch: 98/1     Loss: 0.0025 ± 0.0010
INFO:adv-cli.py:Epoch: 98/1     Fact Loss: 354.4487
INFO:adv-cli.py:Epoch: 99/1     Loss: 0.0024 ± 0.0011
INFO:adv-cli.py:Epoch: 99/1     Fact Loss: 339.9706
INFO:adv-cli.py:Epoch: 100/1    Loss: 0.0023 ± 0.0008
INFO:adv-cli.py:Epoch: 100/1    Fact Loss: 327.1474
[..]

```

[1] Bordes, A. et al. - [Translating Embeddings for Modeling Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) - NIPS 2013
