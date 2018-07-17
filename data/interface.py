import natural_logic_model as nlm
import util

print("Input a premise sentence and hypothesis sentence of the form:\n Determiner (Adjective) Noun (does not) Verb Determiner Adjective Noun \n Make sure you conjugate to the present tense and use vocabulary from the files in the Data folder\n You can also combine two simple sentences of that form with: or, and, if...then)
while True:
    premise = util.parse_sentence(input("Enter a premise sentence"))
    while premise == None:
        premise = util.parse_sentence(input("There was some issue with the entered premise\n Enter a premise sentence:"))
    hypothesis = util.parse_sentence(input("Enter a hypothesis sentence"))
    while hypothesis == None:
        hypothesis = util.parse_sentence(input("There was some issue with the entered premise\n Enter a premise sentence:"))
    if len(premise) == 1:
        label = nlm.get_label(nlm.compute_simple_relation(premise[0], hypothesis[0]))
    else:
        label = nlm.get_label(nlm.compute_boolean_relation(premise[0],premise[1],premise[2], hypothesis[0], hypothesis[1],hypothesis[2]))
    print("The label for your premise and hypothesis is:", label)
    print("You can now input a new premise and hypothesis")
