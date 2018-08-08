import generate_data as gd
import random
if __name__ == '__main__':
    train_size = 500000
    val_size = 10000
    test_size = 10000
    data, _, _ = gd.process_data(1.0)
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,train_size, data, simple_sampling = "level 2", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment2_level21.train")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,val_size, data, simple_sampling = "level 2", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment2_level21.val")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,test_size, data, simple_sampling = "level 2", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment2_level21.test")
