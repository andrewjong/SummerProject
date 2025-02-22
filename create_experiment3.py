import generate_data as gd
import random
if __name__ == '__main__':
    train_size = 500000
    val_size = 10000
    test_size = 10000
    train_data, val_data, test_data= gd.process_data(0.6)
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size, 0, train_data, simple_sampling = "level 0", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment3_level0.train")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", val_size, 0, val_data, simple_sampling = "level 0", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment3_level0.val")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", test_size, 0, test_data, simple_sampling = "level 0", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment3_level0.test")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size, 0, train_data, simple_sampling = "level 1", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment3_level1.train")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", val_size, 0, val_data, simple_sampling = "level 1", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment3_level1.val")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", test_size, 0, test_data, simple_sampling = "level 1", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment3_level1.test")
