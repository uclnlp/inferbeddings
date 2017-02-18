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

Example - Embedding the WN18 Knowledge Graph using Complex Embeddings:

```
$ ./bin/adv-cli.py --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model ComplEx --similarity dot --margin 5 --embedding-size 100 --nb-epochs 100
INFO:adv-cli.py:Command line: --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt 
--lr 0.1 --model ComplEx --similarity dot --margin 5 --embedding-size 100 --nb-epochs 100
INFO:adv-cli.py:#Training Triples: 141442, #Validation Triples: 5000, #Test Triples: 5000
INFO:adv-cli.py:#Entities: 40943        #Predicates: 18
INFO:adv-cli.py:Samples: 141442, no. batches: 10 -> batch size: 14145
INFO:adv-cli.py:Epoch: 1/1      Loss: 9.6925 ± 0.3596
INFO:adv-cli.py:Epoch: 1/1      Fact Loss: 1370931.7656
INFO:adv-cli.py:Epoch: 2/1      Loss: 5.6142 ± 1.1346
INFO:adv-cli.py:Epoch: 2/1      Fact Loss: 794099.1484
INFO:adv-cli.py:Epoch: 3/1      Loss: 1.9155 ± 0.2875
INFO:adv-cli.py:Epoch: 3/1      Fact Loss: 270929.2051
INFO:adv-cli.py:Epoch: 4/1      Loss: 0.7533 ± 0.0712
INFO:adv-cli.py:Epoch: 4/1      Fact Loss: 106553.3623
INFO:adv-cli.py:Epoch: 5/1      Loss: 0.3941 ± 0.0196
INFO:adv-cli.py:Epoch: 5/1      Fact Loss: 55737.0645
[..]
INFO:adv-cli.py:Epoch: 97/1     Loss: 0.0090 ± 0.0018
INFO:adv-cli.py:Epoch: 97/1     Fact Loss: 1267.5016
INFO:adv-cli.py:Epoch: 98/1     Loss: 0.0095 ± 0.0018
INFO:adv-cli.py:Epoch: 98/1     Fact Loss: 1347.9553
INFO:adv-cli.py:Epoch: 99/1     Loss: 0.0094 ± 0.0019
INFO:adv-cli.py:Epoch: 99/1     Fact Loss: 1332.0370
INFO:adv-cli.py:Epoch: 100/1    Loss: 0.0088 ± 0.0014
INFO:adv-cli.py:Epoch: 100/1    Fact Loss: 1243.2720
[..]
INFO:inferbeddings.evaluation.base:### MICRO (test filtered):
INFO:inferbeddings.evaluation.base:     -- left   >> mean: 555.7002, median: 1.0, mrr: 0.923, hits@10: 94.48%
INFO:inferbeddings.evaluation.base:     -- right  >> mean: 554.135, median: 1.0, mrr: 0.924, hits@10: 94.54%
INFO:inferbeddings.evaluation.base:     -- global >> mean: 554.9176, median: 1.0, mrr: 0.924, hits@10: 94.51%
```

Example - Embedding the WN18 Knowledge Graph using Translating Embeddings:

```bash
$ ./bin/adv-cli.py --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model TransE --similarity l1 --margin 2 --embedding-size 50 --nb-epochs 1000
INFO:adv-cli.py:Command line: --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model TransE --similarity l1 --margin 2 --embedding-size 50 --nb-epochs 1000
INFO:adv-cli.py:#Training Triples: 141442, #Validation Triples: 5000, #Test Triples: 5000
INFO:adv-cli.py:#Entities: 40943        #Predicates: 18
INFO:adv-cli.py:Samples: 141442, no. batches: 10 -> batch size: 14145
INFO:adv-cli.py:Epoch: 1/1      Loss: 3.6197 ± 0.5006
INFO:adv-cli.py:Epoch: 1/1      Fact Loss: 511980.6953
INFO:adv-cli.py:Epoch: 2/1      Loss: 2.0904 ± 0.0947
INFO:adv-cli.py:Epoch: 2/1      Fact Loss: 295677.3770
INFO:adv-cli.py:Epoch: 3/1      Loss: 1.3305 ± 0.0399
INFO:adv-cli.py:Epoch: 3/1      Fact Loss: 188185.9883
INFO:adv-cli.py:Epoch: 4/1      Loss: 0.9019 ± 0.0126
INFO:adv-cli.py:Epoch: 4/1      Fact Loss: 127561.4561
INFO:adv-cli.py:Epoch: 5/1      Loss: 0.6394 ± 0.0117
INFO:adv-cli.py:Epoch: 5/1      Fact Loss: 90443.4600
[..]
INFO:adv-cli.py:Epoch: 995/1    Loss: 0.0030 ± 0.0004
INFO:adv-cli.py:Epoch: 995/1    Fact Loss: 428.5829
INFO:adv-cli.py:Epoch: 996/1    Loss: 0.0030 ± 0.0009
INFO:adv-cli.py:Epoch: 996/1    Fact Loss: 427.0264
INFO:adv-cli.py:Epoch: 997/1    Loss: 0.0030 ± 0.0004
INFO:adv-cli.py:Epoch: 997/1    Fact Loss: 428.6463
INFO:adv-cli.py:Epoch: 998/1    Loss: 0.0030 ± 0.0005
INFO:adv-cli.py:Epoch: 998/1    Fact Loss: 422.2098
INFO:adv-cli.py:Epoch: 999/1    Loss: 0.0030 ± 0.0008
INFO:adv-cli.py:Epoch: 999/1    Fact Loss: 419.9970
INFO:adv-cli.py:Epoch: 1000/1   Loss: 0.0027 ± 0.0007
INFO:adv-cli.py:Epoch: 1000/1   Fact Loss: 378.9356
[..]
INFO:inferbeddings.evaluation.base:### MICRO (valid filtered):
INFO:inferbeddings.evaluation.base:     -- left   >> mean: 456.7884, median: 2.0, mrr: 0.472, hits@10: 91.74%
INFO:inferbeddings.evaluation.base:     -- right  >> mean: 457.314, median: 2.0, mrr: 0.481, hits@10: 93.58%
INFO:inferbeddings.evaluation.base:     -- global >> mean: 457.0512, median: 2.0, mrr: 0.476, hits@10: 92.66%
[..]
INFO:inferbeddings.evaluation.base:### MICRO (test filtered):
INFO:inferbeddings.evaluation.base:     -- left   >> mean: 384.1844, median: 2.0, mrr: 0.468, hits@10: 92.38%
INFO:inferbeddings.evaluation.base:     -- right  >> mean: 396.3276, median: 2.0, mrr: 0.479, hits@10: 93.56%
INFO:inferbeddings.evaluation.base:     -- global >> mean: 390.256, median: 2.0, mrr: 0.474, hits@10: 92.97%
```

[1] Bordes, A. et al. - [Translating Embeddings for Modeling Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) - NIPS 2013
