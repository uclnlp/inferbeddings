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
$ ./bin/adv-cli.py --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model ComplEx --similarity dot --margin 5 --embedding-size 100 --nb-epochs 1000
INFO:triples-cli.py:#Training Triples: 141442   #Validation Triples: 5000       #Test Triples: 5000
INFO:triples-cli.py:#Entities: 40943    #Predicates: 18
[..]
INFO:triples-cli.py:Epoch: 1    Loss: 2.3996 ± 0.1164
INFO:triples-cli.py:Epoch: 2    Loss: 1.3897 ± 0.3836
INFO:triples-cli.py:Epoch: 3    Loss: 0.474 ± 0.1532
INFO:triples-cli.py:Epoch: 4    Loss: 0.1566 ± 0.04
[..]
INFO:triples-cli.py:Epoch: 997  Loss: 0.0011 ± 0.0003
INFO:triples-cli.py:Epoch: 998  Loss: 0.0011 ± 0.0002
INFO:triples-cli.py:Epoch: 999  Loss: 0.001 ± 0.0003
INFO:triples-cli.py:Epoch: 1000 Loss: 0.001 ± 0.0002
[..]
INFO:root:### MICRO (valid raw):
INFO:root:      -- left   >> mean: 591.449, median: 1.0, mrr: 0.629, hits@10: 82.84%
INFO:root:      -- right  >> mean: 592.4256, median: 1.0, mrr: 0.617, hits@10: 82.06%
INFO:root:      -- global >> mean: 591.9373, median: 1.0, mrr: 0.623, hits@10: 82.45%
INFO:root:### MICRO (valid filtered):
INFO:root:      -- left   >> mean: 580.3538, median: 1.0, mrr: 0.931, hits@10: 94.72%
INFO:root:      -- right  >> mean: 581.1314, median: 1.0, mrr: 0.933, hits@10: 94.66%
INFO:root:      -- global >> mean: 580.7426, median: 1.0, mrr: 0.932, hits@10: 94.69%
INFO:root:### MICRO (test raw):
INFO:root:      -- left   >> mean: 503.482, median: 2.0, mrr: 0.608, hits@10: 82.1%
INFO:root:      -- right  >> mean: 501.257, median: 1.0, mrr: 0.633, hits@10: 83.22%
INFO:root:      -- global >> mean: 502.3695, median: 1.0, mrr: 0.62, hits@10: 82.66%
INFO:root:### MICRO (test filtered):
INFO:root:      -- left   >> mean: 491.47, median: 1.0, mrr: 0.932, hits@10: 94.54%
INFO:root:      -- right  >> mean: 490.4708, median: 1.0, mrr: 0.932, hits@10: 94.52%
INFO:root:      -- global >> mean: 490.9704, median: 1.0, mrr: 0.932, hits@10: 94.53%
```

[1] Bordes, A. et al. - [Translating Embeddings for Modeling Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) - NIPS 2013