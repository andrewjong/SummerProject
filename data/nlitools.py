import generate_data as gd
import json

def compose_relations(x,y):
    return relation_composition[(x,y)]

def strong_composition(signature1, signature2, relation1, relation2):
    composition1 = compose_relations(signature1[relation1], signature2[relation2])
    composition2 = compose_relations(signature2[relation2], signature1[relation1])
    if composition1 == "independence":
        return composition2
    if composition2 != "independence" and composition1 != composition2:
        print("This shouldn't happen", composition1, composition2)
    return composition1

relations = ["equivalence", "entails", "reverse entails", "contradiction", "cover", "alternation", "independence"]
relations2 = ["equivalence", "entails", "reverse entails", "contradiction", "cover", "alternation", "independence"]
relation_composition= dict()
for r in relations:
    for r2 in relations2:
        relation_composition[(r,r2)] = "independence"
for r in relations:
    relation_composition[("equivalence", r)] = r
    relation_composition[(r,"equivalence")] = r
relation_composition[("entails", "entails")] = "entails"
relation_composition[("entails", "contradiction")] = "alternation"
relation_composition[("entails", "alternation")] = "alternation"
relation_composition[("reverse entails", "reverse entails")] = "reverse entails"
relation_composition[("reverse entails", "contradiction")] = "cover"
relation_composition[("reverse entails", "cover")] = "cover"
relation_composition[("contradiction", "entails")] = "cover"
relation_composition[("contradiction", "reverse entails")] = "alternation"
relation_composition[("contradiction", "contradiction")] = "equivalence"
relation_composition[("contradiction", "cover")] = "reverse entails"
relation_composition[("contradiction", "alternation")] = "entails"
relation_composition[("alternation", "reverse entails")] = "alternation"
relation_composition[("alternation", "contradiction")] = "entails"
relation_composition[("alternation", "cover")] = "entails"
relation_composition[("cover", "entails")] = "cover"
relation_composition[("cover", "contradiction")] = "reverse entails"
relation_composition[("cover", "alternation")] = " reverse entails"
negation_signature = {"equivalence":"equivalence", "entails":"reverse entails", "reverse entails":"entails", "contradiction":"contradiction", "cover":"alternation", "alternation":"cover", "independence":"independence"}
emptystring_signature = {"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "contradiction":"contradiction", "cover":"cover", "alternation":"alternation", "independence":"independence"}
compose_contradiction_signature = {r:relation_composition[(r, "contradiction")] for r in relations }
determiner_signatures = dict()
symmetric_relation = {"equivalence":"equivalence", "entails":"reverse entails", "reverse entails":"entails", "contradiction":"contradiction", "cover":"cover", "alternation":"alternation", "independence":"independence"}
determiner_signatures[("some","some")] =(
{"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "independence":"independence"},
{"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "contradiction":"cover", "cover":"cover", "alternation":"independence", "independence":"independence"}
)
determiner_signatures[("every","every")] =(
{"equivalence":"equivalence", "entails":"reverse entails", "reverse entails":"entails", "independence":"independence"},
{"equivalence":"equivalence", "entails":"entails", "reverse entails":"reverse entails", "contradiction":"alternation", "cover":"independence", "alternation":"alternation", "independence":"independence"}
)
for key in determiner_signatures:
    signature1, signature2 = determiner_signatures[key]
    new_signature = dict()
    for key1 in signature1:
        for key2 in signature2:
            new_signature[(key1, key2)] = strong_composition(signature1, signature2, key1, key2)
    determiner_signatures[key] = new_signature

new_signature = dict()
for relation1 in ["equivalence", "entails", "reverse entails", "independence"]:
    for relation2 in relations:
        if (relation2 == "equivalence" or relation2 == "reverse entails") and relation1 != "independence":
            new_signature[(relation1, relation2)] = "reverse entails"
        else:
            new_signature[(relation1, relation2)] = "independence"
determiner_signatures[("some","every")] = new_signature
determiner_signatures[("some","every")][("entails", "contradiction")] = "alternation"
determiner_signatures[("some","every")][("entails", "alternation")] = "alternation"
determiner_signatures[("some","every")][("equivalence", "alternation")] = "alternation"
determiner_signatures[("some","every")][("equivalence", "contradiction")] = "contradiction"
determiner_signatures[("some","every")][("equivalence", "cover")] = "cover"
determiner_signatures[("some","every")][("reverse entails", "cover")] = "cover"
determiner_signatures[("some","every")][("contradiction", "cover")] = "cover"

new_signature = dict()
for key in determiner_signatures[("some", "every")]:
    new_signature[(symmetric_relation[key[0]], symmetric_relation[key[1]])] = symmetric_relation[determiner_signatures["some", "every"][key]]
determiner_signatures[("every", "some")] = new_signature

def compose_signatures(f,g):
    h = dict()
    for r in f:
        h[r] = g[f[r]]
    return h


def standard_lexical_merge(x,y):
    if  x == y:
        return "equivalence"
    if x == "":
        return "reverse entails"
    if y == "":
        return "entails"
    return "independence"

def determiner_merge(determiner1,determiner2):
    return determiner_signatures[(determiner1,determiner2)]

def negation_merge(negation1, negation2):
    if negation1 == negation2 and not negation2:
        return emptystring_signature
    if negation1 == negation2 and negation2 :
        return negation_signature
    if not negation1:
        return compose_contradiction_signature
    if negation1:
        return compose_signatures(negation_signature, compose_contradiction_signature)
    print("oh fuck")

def standard_phrase(relation1, relation2):
    if relation2 == "equivalence":
        return relation1
    return "independence"

def determiner_phrase(signature, relation1, relation2):
    return signature[(relation1,relation2)]


def negation_phrase(negation_signature, relation):
    return negation_signature[relation]

def get_final_label(relation):
    if relation in ["cover", "independence", "reverse entails"]:
        return "permits"
    if relation in ["entails", "equivalence"]:
        return "entails"
    if relation in ["alternation", "contradiction"]:
        return "contradicts"

def compute_label(premise, hypothesis):
    #leaves
    subject_negation_signature = negation_merge(premise.subject_negation, hypothesis.subject_negation)
    subject_determiner_signature = determiner_merge(premise.natlog_subject_determiner, hypothesis.natlog_subject_determiner)
    subject_noun_relation = standard_lexical_merge(premise.subject_noun,hypothesis.subject_noun)
    subject_adjective_relation = standard_lexical_merge(premise.subject_adjective,hypothesis.subject_adjective)
    verb_negation_signature = negation_merge(premise.verb_negation, hypothesis.verb_negation)
    verb_relation = standard_lexical_merge(premise.verb,hypothesis.verb)
    adverb_relation = standard_lexical_merge(premise.adverb,hypothesis.adverb)
    object_negation_signature = negation_merge(premise.object_negation, hypothesis.object_negation)
    object_determiner_signature = determiner_merge(premise.natlog_object_determiner, hypothesis.natlog_object_determiner)
    object_noun_relation = standard_lexical_merge(premise.object_noun,hypothesis.object_noun)
    object_adjective_relation = standard_lexical_merge(premise.object_adjective,hypothesis.object_adjective)

    #first layer nodes
    VP_relation = standard_phrase(adverb_relation, verb_relation)
    object_NP_relation = standard_phrase(object_adjective_relation, object_noun_relation)
    subject_NP_relation = standard_phrase(subject_adjective_relation, subject_noun_relation)

    object_DP_relation = determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)

    object_negDP_relation = negation_phrase(object_negation_signature, object_DP_relation)

    negverb_relation = negation_phrase(verb_negation_signature, object_negDP_relation)

    subject_DP_relation = determiner_phrase(subject_determiner_signature, subject_NP_relation, negverb_relation)

    subject_NegDP_relation = negation_phrase(subject_negation_signature, object_DP_relation)
    return get_final_label(subject_NegDP_relation)

data, _, _ = gd.process_data(1.0)
with open("simple_solutions2", 'r') as f:
    solutions= json.loads(f.read())
count = 0
count2 = 0
for encoding in solutions:
    premise, hypothesis = gd.encoding_to_example(data, json.loads(encoding))
    count += 1
    if solutions[encoding] != compute_label(premise, hypothesis):
        #print(solutions[encoding])
        #print(compute_label(premise, hypothesis))
        #print(premise.string)
        #print(hypothesis.string)
        count2 += 1
print(count, count2)
