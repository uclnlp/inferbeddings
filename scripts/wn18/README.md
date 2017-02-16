# WN18 - Experiments

- UCL_WN18_adv_v1.py - TransE (100 epochs)
- UCL_WN18_adv_v1.1.py - Variant of UCL_WN18_adv_v1.py where the margin is varied in {2, 5, 10}

- UCL_WN18_adv_v2.py - DistMult (100 epochs)
- UCL_WN18_adv_v2.1.py - Variant of UCL_WN18_adv_v2.py where the margin is varied in {2, 5, 10}

- UCL_WN18_adv_v3.py - ComplEx (100 epochs)
- UCL_WN18_adv_v3.1.py - Variant of UCL_WN18_adv_v3.py where the margin is varied in {2, 5, 10}

- UCL_WN18_adv_corrupt_relations_v1.py - TransE, DistMult and ComplEx with --corrupt-relations flag
Grid:
        margin=[1, 2, 5, 10],
        embedding_size=[20, 50, 100, 150, 200],
        adv_epochs=[0, 10],
        adv_weight=[0, 1, 10, 100, 1000, 10000],
        adv_batch_size=[1, 10, 100]

- UCL_WN18_adv_logistic_v1.py - TransE, DistMult and ComplEx with --loss logistic flag
Grid:
        margin=[1],
        embedding_size=[20, 50, 100, 150, 200],
        adv_epochs=[0, 1, 10],
        disc_epochs=[1, 10],
        adv_weight=[0, 1, 10, 100, 1000, 10000],
        adv_batch_size=[1, 10, 100]
