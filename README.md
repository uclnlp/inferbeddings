# Inferbeddings 

[![wercker status](https://app.wercker.com/status/47cfd21ce8315fb361ac5b4247ce280a/s/master "wercker status")](https://app.wercker.com/project/byKey/47cfd21ce8315fb361ac5b4247ce280a)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/uclmr/inferbeddings/blob/master/LICENSE)

Source code used for the experiments in: 

Pasquale Minervini, Thomas Demeester, Tim Rocktäschel, Sebastian Riedel: [Adversarial Sets for Regularising Neural Link Predictors](http://auai.org/uai2017/proceedings/papers/306.pdf) UAI 2017

```
@inproceedings{DBLP:conf/uai/MinerviniDRR17,
  author    = {Pasquale Minervini and
               Thomas Demeester and
               Tim Rockt{\"{a}}schel and
               Sebastian Riedel},
  title     = {Adversarial Sets for Regularising Neural Link Predictors},
  booktitle = {Proceedings of the Thirty-Third Conference on Uncertainty in Artificial
               Intelligence, {UAI} 2017},
  year      = {2017},
  crossref  = {DBLP:conf/uai/2017}
}
@proceedings{DBLP:conf/uai/2017,
  editor    = {Gal Elidan and
               Kristian Kersting and
               Alexander T. Ihler},
  title     = {Proceedings of the Thirty-Third Conference on Uncertainty in Artificial
               Intelligence, {UAI} 2017},
  publisher = {{AUAI} Press},
  year      = {2017}
}
```

Injecting Background Knowledge in Neural Models via Adversarial Set Regularisation.


Usage:

```
$ python3 setup.py install --user
running install
running bdist_egg
running egg_info
[...]
$ ./bin/kbp-cli.py -h 
INFO:kbp-cli.py:Command line: -h
usage: Rule Injection via Adversarial Training [-h] --train TRAIN [--valid VALID] [--test TEST] [--debug] [--debug-scores DEBUG_SCORES [DEBUG_SCORES ...]] [--debug-embeddings DEBUG_EMBEDDINGS]
                                               [--debug-results] [--lr LR] [--initial-accumulator-value INITIAL_ACCUMULATOR_VALUE] [--nb-batches NB_BATCHES] [--nb-epochs NB_EPOCHS] [--model MODEL]
                                               [--similarity SIMILARITY] [--loss LOSS] [--pairwise-loss PAIRWISE_LOSS] [--corrupt-relations] [--margin MARGIN] [--embedding-size EMBEDDING_SIZE]
                                               [--predicate-embedding-size PREDICATE_EMBEDDING_SIZE] [--hidden-size HIDDEN_SIZE] [--unit-cube]
                                               [--all-one-entities ALL_ONE_ENTITIES [ALL_ONE_ENTITIES ...]] [--predicate-l2 PREDICATE_L2] [--predicate-norm PREDICATE_NORM] [--auc] [--seed SEED]
                                               [--clauses CLAUSES] [--sar-weight SAR_WEIGHT] [--sar-similarity SAR_SIMILARITY] [--adv-lr ADV_LR] [--adversary-epochs ADVERSARY_EPOCHS]
                                               [--discriminator-epochs DISCRIMINATOR_EPOCHS] [--adv-weight ADV_WEIGHT] [--adv-margin ADV_MARGIN] [--adv-batch-size ADV_BATCH_SIZE] [--adv-init-ground]
                                               [--adv-ground-samples ADV_GROUND_SAMPLES] [--adv-ground-tol ADV_GROUND_TOL] [--adv-pooling ADV_POOLING] [--subsample-size SUBSAMPLE_SIZE]
                                               [--head-subsample-size HEAD_SUBSAMPLE_SIZE] [--materialize] [--save SAVE]

optional arguments:
  -h, --help                                                                                  show this help message and exit
  --train TRAIN, -t TRAIN
  --valid VALID, -v VALID
  --test TEST, -T TEST
  --debug, -D                                                                                 Debug flag
  --debug-scores DEBUG_SCORES [DEBUG_SCORES ...]                                              List of files containing triples we want to compute the score of
  --debug-embeddings DEBUG_EMBEDDINGS
  --debug-results                                                                             Report fine-grained ranking results
  --lr LR, -l LR
  --initial-accumulator-value INITIAL_ACCUMULATOR_VALUE
  --nb-batches NB_BATCHES, -b NB_BATCHES
  --nb-epochs NB_EPOCHS, -e NB_EPOCHS
  --model MODEL, -m MODEL                                                                     Model
  --similarity SIMILARITY, -s SIMILARITY                                                      Similarity function
  --loss LOSS                                                                                 Loss function
  --pairwise-loss PAIRWISE_LOSS                                                               Pairwise loss function
  --corrupt-relations                                                                         Also corrupt the relation of each training triple for generating negative examples
  --margin MARGIN, -M MARGIN                                                                  Margin
  --embedding-size EMBEDDING_SIZE, --entity-embedding-size EMBEDDING_SIZE, -k EMBEDDING_SIZE  Entity embedding size
  --predicate-embedding-size PREDICATE_EMBEDDING_SIZE, -p PREDICATE_EMBEDDING_SIZE            Predicate embedding size
  --hidden-size HIDDEN_SIZE, -H HIDDEN_SIZE                                                   Size of the hidden layer (if necessary, e.g. ER-MLP)
  --unit-cube                                                                                 Project all entity embeddings on the unit cube (rather than the unit ball)
  --all-one-entities ALL_ONE_ENTITIES [ALL_ONE_ENTITIES ...]                                  Entities with all-one entity embeddings
  --predicate-l2 PREDICATE_L2                                                                 Weight of the L2 regularization term on the predicate embeddings
  --predicate-norm PREDICATE_NORM                                                             Norm of the predicate embeddings
  --auc, -a                                                                                   Measure the predictive accuracy using AUC-PR and AUC-ROC
  --seed SEED, -S SEED                                                                        Seed for the PRNG
  --clauses CLAUSES, -c CLAUSES                                                               File containing background knowledge expressed as Horn clauses
  --sar-weight SAR_WEIGHT                                                                     Schema-Aware Regularization, regularizer weight
  --sar-similarity SAR_SIMILARITY                                                             Schema-Aware Regularization, similarity measure
  --adv-lr ADV_LR, -L ADV_LR                                                                  Adversary learning rate
  --adversary-epochs ADVERSARY_EPOCHS                                                         Adversary - number of training epochs
  --discriminator-epochs DISCRIMINATOR_EPOCHS                                                 Discriminator - number of training epochs
  --adv-weight ADV_WEIGHT, -W ADV_WEIGHT                                                      Adversary weight
  --adv-margin ADV_MARGIN                                                                     Adversary margin
  --adv-batch-size ADV_BATCH_SIZE                                                             Size of the batch of adversarial examples to use
  --adv-init-ground                                                                           Initialize adversarial embeddings using real entity embeddings
  --adv-ground-samples ADV_GROUND_SAMPLES                                                     Number of ground samples on which to compute the ground loss
  --adv-ground-tol ADV_GROUND_TOL, --adv-ground-tolerance ADV_GROUND_TOL                      Epsilon-tolerance when calculating the ground loss
  --adv-pooling ADV_POOLING                                                                   Pooling method used for aggregating adversarial losses (sum, mean, max, logsumexp)
  --subsample-size SUBSAMPLE_SIZE                                                             Fraction of training facts to use during training (e.g. 0.1)
  --head-subsample-size HEAD_SUBSAMPLE_SIZE                                                   Fraction of training facts to use during training (e.g. 0.1)
  --materialize                                                                               Materialize all facts using clauses and logical inference
  --save SAVE                                                                                 Path for saving the serialized model

```

If the parameter `--adv-lr` is not specified, the method does not perform any adversarial training -- i.e. it simply trains the Knowledge Graph Embedding models by minimizing a standard pairwise loss in, such as the margin-based ranking loss in [1].

Example - Embedding the WN18 Knowledge Graph using Complex Embeddings:

```
$ ./bin/kbp-cli.py --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model ComplEx --similarity dot --margin 5 --embedding-size 100 --nb-epochs 100
INFO:kbp-cli.py:Command line: --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model ComplEx --similarity dot --margin 5 --embedding-size 100 --nb-epochs 100
INFO:kbp-cli.py:#Training Triples: 141442, #Validation Triples: 5000, #Test Triples: 5000
INFO:kbp-cli.py:#Entities: 40943        #Predicates: 18
INFO:kbp-cli.py:Samples: 141442, no. batches: 10 -> batch size: 14145
INFO:kbp-cli.py:Epoch: 1/1      Loss: 9.9205 ± 0.0819
INFO:kbp-cli.py:Epoch: 1/1      Fact Loss: 1403175.1875
INFO:kbp-cli.py:Epoch: 2/1      Loss: 9.1740 ± 0.1622
INFO:kbp-cli.py:Epoch: 2/1      Fact Loss: 1297591.9766
INFO:kbp-cli.py:Epoch: 3/1      Loss: 8.3536 ± 0.0911
INFO:kbp-cli.py:Epoch: 3/1      Fact Loss: 1181545.5312
INFO:kbp-cli.py:Epoch: 4/1      Loss: 7.7184 ± 0.0576
INFO:kbp-cli.py:Epoch: 4/1      Fact Loss: 1091707.6953
INFO:kbp-cli.py:Epoch: 5/1      Loss: 7.1793 ± 0.0403
INFO:kbp-cli.py:Epoch: 5/1      Fact Loss: 1015459.9766
INFO:kbp-cli.py:Epoch: 6/1      Loss: 6.7136 ± 0.0240
INFO:kbp-cli.py:Epoch: 6/1      Fact Loss: 949580.1484
INFO:kbp-cli.py:Epoch: 7/1      Loss: 6.3037 ± 0.0203
INFO:kbp-cli.py:Epoch: 7/1      Fact Loss: 891614.3750
INFO:kbp-cli.py:Epoch: 8/1      Loss: 5.9400 ± 0.0131
INFO:kbp-cli.py:Epoch: 8/1      Fact Loss: 840160.1172
INFO:kbp-cli.py:Epoch: 9/1      Loss: 5.6087 ± 0.0192
INFO:kbp-cli.py:Epoch: 9/1      Fact Loss: 793311.0703
INFO:kbp-cli.py:Epoch: 10/1     Loss: 5.3024 ± 0.0215
INFO:kbp-cli.py:Epoch: 10/1     Fact Loss: 749982.7891
[..]
INFO:kbp-cli.py:Epoch: 91/1     Loss: 0.2401 ± 0.0085
INFO:kbp-cli.py:Epoch: 91/1     Fact Loss: 33953.7292
INFO:kbp-cli.py:Epoch: 92/1     Loss: 0.2347 ± 0.0053
INFO:kbp-cli.py:Epoch: 92/1     Fact Loss: 33194.0295
INFO:kbp-cli.py:Epoch: 93/1     Loss: 0.2315 ± 0.0077
INFO:kbp-cli.py:Epoch: 93/1     Fact Loss: 32742.8557
INFO:kbp-cli.py:Epoch: 94/1     Loss: 0.2280 ± 0.0066
INFO:kbp-cli.py:Epoch: 94/1     Fact Loss: 32254.6658
INFO:kbp-cli.py:Epoch: 95/1     Loss: 0.2221 ± 0.0078
INFO:kbp-cli.py:Epoch: 95/1     Fact Loss: 31411.6094
INFO:kbp-cli.py:Epoch: 96/1     Loss: 0.2181 ± 0.0055
INFO:kbp-cli.py:Epoch: 96/1     Fact Loss: 30844.9131
INFO:kbp-cli.py:Epoch: 97/1     Loss: 0.2119 ± 0.0073
INFO:kbp-cli.py:Epoch: 97/1     Fact Loss: 29971.1079
INFO:kbp-cli.py:Epoch: 98/1     Loss: 0.2107 ± 0.0061
INFO:kbp-cli.py:Epoch: 98/1     Fact Loss: 29807.2014
INFO:kbp-cli.py:Epoch: 99/1     Loss: 0.2081 ± 0.0054
INFO:kbp-cli.py:Epoch: 99/1     Fact Loss: 29432.5969
INFO:kbp-cli.py:Epoch: 100/1    Loss: 0.2042 ± 0.0067
INFO:kbp-cli.py:Epoch: 100/1    Fact Loss: 28878.2566
[..]
INFO:inferbeddings.evaluation.base:### MICRO (test filtered):
INFO:inferbeddings.evaluation.base:     -- left   >> mean: 438.0892, median: 1.0, mrr: 0.857, hits@10: 92.32%
INFO:inferbeddings.evaluation.base:     -- right  >> mean: 441.4096, median: 1.0, mrr: 0.868, hits@10: 92.38%
INFO:inferbeddings.evaluation.base:     -- global >> mean: 439.7494, median: 1.0, mrr: 0.862, hits@10: 92.35%
```

Example - Embedding the WN18 Knowledge Graph using Translating Embeddings:

```bash
$ ./bin/kbp-cli.py --train data/wn18/wordnet-mlj12-train.txt --valid data/wn18/wordnet-mlj12-valid.txt --test data/wn18/wordnet-mlj12-test.txt --lr 0.1 --model TransE --similarity l1 --margin 2 --embedding-size 50 --nb-epochs 1000
INFO:kbp-cli.py:#Training Triples: 141442, #Validation Triples: 5000, #Test Triples: 5000
INFO:kbp-cli.py:#Entities: 40943        #Predicates: 18
INFO:kbp-cli.py:Samples: 141442, no. batches: 10 -> batch size: 14145
INFO:kbp-cli.py:Epoch: 1/1      Loss: 3.3778 ± 0.5889
INFO:kbp-cli.py:Epoch: 1/1      Fact Loss: 477762.2461
INFO:kbp-cli.py:Epoch: 2/1      Loss: 1.3837 ± 0.1561
INFO:kbp-cli.py:Epoch: 2/1      Fact Loss: 195715.7637
INFO:kbp-cli.py:Epoch: 3/1      Loss: 0.5752 ± 0.0353
INFO:kbp-cli.py:Epoch: 3/1      Fact Loss: 81351.6055
INFO:kbp-cli.py:Epoch: 4/1      Loss: 0.2984 ± 0.0071
INFO:kbp-cli.py:Epoch: 4/1      Fact Loss: 42206.5698
INFO:kbp-cli.py:Epoch: 5/1      Loss: 0.1842 ± 0.0028
INFO:kbp-cli.py:Epoch: 5/1      Fact Loss: 26058.0952
INFO:kbp-cli.py:Epoch: 6/1      Loss: 0.1287 ± 0.0017
INFO:kbp-cli.py:Epoch: 6/1      Fact Loss: 18210.4518
INFO:kbp-cli.py:Epoch: 7/1      Loss: 0.0980 ± 0.0023
INFO:kbp-cli.py:Epoch: 7/1      Fact Loss: 13866.0588
INFO:kbp-cli.py:Epoch: 8/1      Loss: 0.0795 ± 0.0034
INFO:kbp-cli.py:Epoch: 8/1      Fact Loss: 11243.2173
INFO:kbp-cli.py:Epoch: 9/1      Loss: 0.0653 ± 0.0019
INFO:kbp-cli.py:Epoch: 9/1      Fact Loss: 9239.1135
INFO:kbp-cli.py:Epoch: 10/1     Loss: 0.0562 ± 0.0026
INFO:kbp-cli.py:Epoch: 10/1     Fact Loss: 7942.8276
[..]
INFO:kbp-cli.py:Epoch: 990/1    Loss: 0.0026 ± 0.0006
INFO:kbp-cli.py:Epoch: 990/1    Fact Loss: 373.8182
INFO:kbp-cli.py:Epoch: 991/1    Loss: 0.0030 ± 0.0007
INFO:kbp-cli.py:Epoch: 991/1    Fact Loss: 419.6469
INFO:kbp-cli.py:Epoch: 992/1    Loss: 0.0025 ± 0.0006
INFO:kbp-cli.py:Epoch: 992/1    Fact Loss: 354.5707
INFO:kbp-cli.py:Epoch: 993/1    Loss: 0.0028 ± 0.0004
INFO:kbp-cli.py:Epoch: 993/1    Fact Loss: 398.7795
INFO:kbp-cli.py:Epoch: 994/1    Loss: 0.0032 ± 0.0006
INFO:kbp-cli.py:Epoch: 994/1    Fact Loss: 450.6929
INFO:kbp-cli.py:Epoch: 995/1    Loss: 0.0028 ± 0.0005
INFO:kbp-cli.py:Epoch: 995/1    Fact Loss: 390.7763
INFO:kbp-cli.py:Epoch: 996/1    Loss: 0.0028 ± 0.0009
INFO:kbp-cli.py:Epoch: 996/1    Fact Loss: 392.9878
INFO:kbp-cli.py:Epoch: 997/1    Loss: 0.0028 ± 0.0005
INFO:kbp-cli.py:Epoch: 997/1    Fact Loss: 391.4912
INFO:kbp-cli.py:Epoch: 998/1    Loss: 0.0026 ± 0.0003
INFO:kbp-cli.py:Epoch: 998/1    Fact Loss: 362.8399
INFO:kbp-cli.py:Epoch: 999/1    Loss: 0.0026 ± 0.0006
INFO:kbp-cli.py:Epoch: 999/1    Fact Loss: 365.5944
INFO:kbp-cli.py:Epoch: 1000/1   Loss: 0.0025 ± 0.0007
INFO:kbp-cli.py:Epoch: 1000/1   Fact Loss: 354.8261
[..]
INFO:inferbeddings.evaluation.base:### MICRO (valid filtered):
INFO:inferbeddings.evaluation.base:     -- left   >> mean: 472.5754, median: 2.0, mrr: 0.491, hits@10: 94.22%
INFO:inferbeddings.evaluation.base:     -- right  >> mean: 480.887, median: 2.0, mrr: 0.495, hits@10: 94.26%
INFO:inferbeddings.evaluation.base:     -- global >> mean: 476.7312, median: 2.0, mrr: 0.493, hits@10: 94.24%
[..]
INFO:inferbeddings.evaluation.base:### MICRO (test filtered):
INFO:inferbeddings.evaluation.base:     -- left   >> mean: 399.6008, median: 2.0, mrr: 0.493, hits@10: 94.24%
INFO:inferbeddings.evaluation.base:     -- right  >> mean: 424.7734, median: 2.0, mrr: 0.493, hits@10: 94.28%
INFO:inferbeddings.evaluation.base:     -- global >> mean: 412.1871, median: 2.0, mrr: 0.493, hits@10: 94.26%
```

[1] Bordes, A. et al. - [Translating Embeddings for Modeling Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) - NIPS 2013
