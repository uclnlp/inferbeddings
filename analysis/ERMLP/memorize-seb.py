from analysis.ERMLP.ermlp import *

from collections import defaultdict


def learn_mapping(train_triples, threshold=1):
    pair_to_relations = defaultdict(set)
    for s, p, o in train_triples:
        pair_to_relations[(s, o)].add(p)

    counts = defaultdict(int)
    for s, o in list(pair_to_relations.keys()):
        for pred1 in pair_to_relations[(s, o)]:
            for pred2 in pair_to_relations[(o, s)]:
                counts[(pred1, pred2)] += 1
    done = set()
    for (pred1, pred2) in sorted(counts.keys(), key=lambda t: -counts[t]):
        if (pred1, pred2) not in done:
            print('"{}":"{}",'.format(pred1, pred2))
            print('"{}":"{}",'.format(pred2, pred1))
            done.update([(pred1, pred2), (pred2, pred1)])
    result = defaultdict(list)
    for (pred1, pred2), count in counts.items():
        if count >= threshold:
            result[pred1].append(pred2)
            result[pred2].append(pred1)
    return result


def evaluate_mapping(mapping, train_triples, valid_triples, test_triples):
    all_triples = train_triples + valid_triples + test_triples
    entity_set = set([s for (s, p, o) in all_triples] + [o for (s, p, o) in all_triples])
    predicate_set = set([p for (s, p, o) in all_triples])

    nb_entities, nb_predicates = len(entity_set), len(predicate_set)

    # print(train_triples[:10])
    # print(mapping)

    train_set = set(train_triples)
    err_corrupt_subj, err_corrupt_obj = [], []
    for s, p, o in valid_triples:
        # if p not in mapping:
        #     print(p)
        if p in mapping and any((o, m, s) in train_set for m in mapping[p]):
            err_corrupt_obj += [1]
            err_corrupt_subj += [1]
        else:
            err_corrupt_obj += [nb_entities]
            err_corrupt_subj += [nb_entities]

    err = err_corrupt_subj + err_corrupt_obj
    print(err)
    print('Mean Rank: {}'.format(np.mean(err)))

    for k in [1, 3, 5, 10]:
        print('Hits@{}: {}'.format(k, np.mean(np.asarray(err) <= k) * 100))


def evaluate_WN18():
    forward_mapping = {
        "_member_of_domain_topic": "_synset_domain_topic_of",
        "_synset_domain_usage_of": "_member_of_domain_usage",
        "_instance_hyponym": "_instance_hypernym",
        "_hyponym": "_hypernym",
        "_member_holonym": "_member_meronym",
        "_synset_domain_region_of": "_member_of_domain_region",
        "_part_of": "_has_part",
        "_member_meronym": "_member_holonym",
        "_hypernym": "_hyponym",
        "_synset_domain_topic_of": "_member_of_domain_topic",
        "_instance_hypernym": "_instance_hyponym",
        "_has_part": "_part_of",
        "_member_of_domain_region": "_synset_domain_region_of",
        "_member_of_domain_usage": "_synset_domain_usage_of",
        "_derivationally_related_form": "_derivationally_related_form",
        "_verb_group": "_verb_group"
    }
    user_mapping = dict(list(forward_mapping.items()) + [(value, key) for key, value in forward_mapping.items()])

    train_triples = read_triples('wn18.triples.train')
    valid_triples = read_triples('wn18.triples.valid')
    test_triples = read_triples('wn18.triples.test')

    learnt_mapping = learn_mapping(train_triples, 100)

    # for key, value in learnt_mapping.items():
    #     print((key, value))
    # print(learnt_mapping)

    evaluate_mapping(learnt_mapping, train_triples, valid_triples, test_triples)


def evaluate_FB122():
    train_triples = read_triples('../../data/guo-emnlp16/data/fb122/fb122_triples.train')
    valid_triples = read_triples('../../data/guo-emnlp16/data/fb122/fb122_triples.valid')
    test_triples = read_triples('../../data/guo-emnlp16/data/fb122/fb122_triples.test')

    learnt_mapping = learn_mapping(train_triples, 2000)

    # for key, value in learnt_mapping.items():
    #     print((key, value))
    # print(learnt_mapping)

    evaluate_mapping(learnt_mapping, train_triples, valid_triples, test_triples)


# evaluate_WN18()
evaluate_FB122()
