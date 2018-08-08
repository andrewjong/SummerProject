import generate_data as gd
import random
if __name__ == '__main__':
    train_size = 500000
    val_size = 10000
    test_size = 10000
    data, _, _ = gd.process_data(1.0)
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data, simple_sampling = "level 1", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment5_level10.train")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions",val_size,val_size, data, simple_sampling = "level 1", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment5_level10.val")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", test_size,test_size, data, simple_sampling = "level 1", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment5_level10.test")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data, simple_sampling = "level 1", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment5_level11.train")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", val_size,val_size, data, simple_sampling = "level 1", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment5_level11.val")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions",test_size, test_size, data, simple_sampling = "level 1", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment5_level11.test")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data, simple_sampling = "level 0", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment5_level00.train")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", val_size,val_size, data, simple_sampling = "level 0", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment5_level00.val")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", test_size,test_size, data, simple_sampling = "level 0", boolean_sampling = "level 0")
    gd.save_data(examples, "experiment5_level00.test")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data, simple_sampling = "level 0", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment5_level01.train")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", val_size,val_size, data, simple_sampling = "level 0", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment5_level01.val")
    examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions",test_size, test_size, data, simple_sampling = "level 0", boolean_sampling = "level 1")
    gd.save_data(examples, "experiment5_level01.test")
