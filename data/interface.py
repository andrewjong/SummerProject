import natural_logic_model as nlm
import util
import generate_data as gd

data, _,_ = gd.process_data(1.0)
print("Input a premise sentence and hypothesis sentence of the form:\n Determiner (Adjective) Noun (does not) Verb Determiner Adjective Noun \n Make sure you conjugate to the present tense and use vocabulary from the files in the Data folder\n You can also combine two simple sentences of that form with: or, and, if...then")
while True:
    premise = util.parse_sentence(data,input("Enter a premise sentence:\n"))
    while premise == None:
        premise = util.parse_sentence(data,input("There was some issue with the entered premise\n Enter a premise sentence:\n"))
    hypothesis = util.parse_sentence(data,input("Enter a hypothesis sentence:\n"))
    while hypothesis == None:
        hypothesis = util.parse_sentence(data,input("There was some issue with the entered premise\n Enter a premise sentence:\n"))
    if len(premise) == 1:
        label = nlm.get_label(nlm.compute_simple_relation(premise[0], hypothesis[0]))
    else:
        label = nlm.get_label(nlm.compute_boolean_relation(premise[0],premise[1],premise[2], hypothesis[0], hypothesis[1],hypothesis[2]))
    print("The label for your premise and hypothesis is: ", label, "\n")
    print("You can now input a new premise and hypothesis\n")
