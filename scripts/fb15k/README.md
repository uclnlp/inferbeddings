# FB15k - Experiments

- UCL_FB15k_adv_v1.py - TransE (100 epochs), rules in data/fb15k/clauses/clauses_0.999.pl
- UCL_FB15k_adv_v1.1.py - Variant of UCL_FB15k_adv_v1.py where the margin is varied in {2, 5, 10}
- UCL_FB15k_adv_v1.2.py - Previous two ones, with rules in data/fb15k/clauses/clauses_0.9_mins=1000_minis=1000.pl

- UCL_FB15k_adv_v2.py - DistMult (100 epochs), rules in data/fb15k/clauses/clauses_0.999.pl
- UCL_FB15k_adv_v2.1.py - Variant of UCL_FB15k_adv_v2.py where the margin is varied in {2, 5, 10}
- UCL_FB15k_adv_v2.2.py - Previous two ones, with rules in data/fb15k/clauses/clauses_0.9_mins=1000_minis=1000.pl

- UCL_FB15k_adv_v3.py - ComplEx (100 epochs), rules in data/fb15k/clauses/clauses_0.999.pl
- UCL_FB15k_adv_v3.1.py - Variant of UCL_FB15k_adv_v3.py where the margin is varied in {2, 5, 10}
- UCL_FB15k_adv_v3.2.py - Previous two ones, with rules in data/fb15k/clauses/clauses_0.9_mins=1000_minis=1000.pl

- UCL_FB15K_adv_corrupt_relations_v1.py - TransE, DistMult and ComplEx with --corrupt-relations flag
Grid:
        clausefile=['clauses_highconf_highsupp.pl', 'clauses_lowconf_highsupp.pl'],
        margin=[1, 2, 5, 10],
        embedding_size=[20, 50, 100, 150, 200],
        adv_epochs=[0, 10],
        adv_weight=[0, 1, 10, 100, 1000, 10000],
        adv_batch_size=[1, 10, 100]

- UCL_FB15K_clauses_v1.py - TransE, DistMult and ComplEx with different clauses sets
Grid:
        margin=[1],
        embedding_size=[100],
        adv_lr=[.1],
        adv_epochs=[0, 1, 10],
        disc_epochs=[1, 10],
        adv_weight=[0, 1, 10, 100, 1000, 10000],
        adv_batch_size=[10]
