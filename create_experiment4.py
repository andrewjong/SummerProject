import generate_data as gd
import random
if __name__ == '__main__':
    train_size = 500000
    val_size = 10000
    test_size = 10000
    data, _, _ = gd.process_data(1.0)
    for ratio in [0.02, 0.04, 0.08]:
        e,c,p = gd.split_dict("simple_solutions", [10]*19)
        ekeys = list(e.keys())
        ckeys = list(c.keys())
        pkeys = list(p.keys())
        ecounts, ccounts, pcounts = gd.get_simple_encoding_counts(data, ekeys, ckeys, pkeys)
        ekeys, ckeys, pkeys, ecounts, ccounts, pcounts = gd.trim_simple_encodings(data,ratio, ekeys, ckeys, pkeys, ecounts, ccounts, pcounts)
        ekeys_and_counts = list(zip(ekeys, ecounts))
        ckeys_and_counts = list(zip(ekeys, ecounts))
        pkeys_and_counts = list(zip(ekeys, ecounts))
        random.shuffle(ekeys_and_counts )
        random.shuffle(ckeys_and_counts )
        random.shuffle(pkeys_and_counts )
        e = ekeys_and_counts[:int(ratio*len(ekeys_and_counts))]
        c = ckeys_and_counts[:int(ratio*len(ckeys_and_counts))]
        p = pkeys_and_counts[:int(ratio*len(pkeys_and_counts))]
        ekeys, ecounts = zip(*e)
        ckeys, ccounts = zip(*c)
        pkeys, pcounts = zip(*p)
        train_keys_and_counts = (ekeys, ckeys, pkeys, ecounts, ccounts, pcounts)
        e = ekeys_and_counts[int(ratio*len(ekeys_and_counts)):int((ratio+0.5*(1 - ratio))*len(ekeys_and_counts))]
        c = ckeys_and_counts[int(ratio*len(ckeys_and_counts)):int((ratio+0.5*(1 - ratio))*len(ckeys_and_counts))]
        p = pkeys_and_counts[int(ratio*len(pkeys_and_counts)):int((ratio+0.5*(1 - ratio))*len(pkeys_and_counts))]
        ekeys, ecounts = zip(*e)
        ckeys, ccounts = zip(*c)
        pkeys, pcounts = zip(*p)
        val_keys_and_counts = (ekeys, ckeys, pkeys, ecounts, ccounts, pcounts)
        e = ekeys_and_counts[int((ratio+0.5*(1 - ratio))*len(ekeys_and_counts)):]
        c = ckeys_and_counts[int((ratio+0.5*(1 - ratio))*len(ckeys_and_counts)):]
        p = pkeys_and_counts[int((ratio+0.5*(1 - ratio))*len(pkeys_and_counts)):]
        ekeys, ecounts = zip(*e)
        ckeys, ccounts = zip(*c)
        pkeys, pcounts = zip(*p)
        test_keys_and_counts = (ekeys, ckeys, pkeys, ecounts, ccounts, pcounts)
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data,simple_sampling = "level 1", boolean_sampling = "level 0", keys_and_counts = train_keys_and_counts)
        gd.save_data(examples, str(ratio)+ "experiment4_level10.train")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,val_size, data,simple_sampling = "level 1", boolean_sampling = "level 0", keys_and_counts = val_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level10.val")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,test_size, data,simple_sampling = "level 1", boolean_sampling = "level 0", keys_and_counts = test_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level10.test")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data,simple_sampling = "level 1", boolean_sampling = "level 1", keys_and_counts = train_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level11.train")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,val_size, data,simple_sampling = "level 1", boolean_sampling = "level 1", keys_and_counts = val_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level11.val")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions",0, test_size, data,simple_sampling = "level 1", boolean_sampling = "level 1", keys_and_counts = test_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level11.test")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data,simple_sampling = "level 0", boolean_sampling = "level 0", keys_and_counts = train_keys_and_counts)
        gd.save_data(examples, str(ratio)+ "experiment4_level00.train")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,val_size, data,simple_sampling = "level 0", boolean_sampling = "level 0", keys_and_counts = val_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level00.val")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,test_size, data,simple_sampling = "level 0", boolean_sampling = "level 0", keys_and_counts = test_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level00.test")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", train_size,train_size, data,simple_sampling = "level 0", boolean_sampling = "level 1", keys_and_counts = train_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level01.train")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions", 0,val_size, data,simple_sampling = "level 0", boolean_sampling = "level 1", keys_and_counts = val_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level01.val")
        examples = gd.generate_balanced_data("simple_solutions", "boolean_solutions",0, test_size, data,simple_sampling = "level 0", boolean_sampling = "level 1", keys_and_counts = test_keys_and_counts)
        gd.save_data(examples, str(ratio)+"experiment4_level01.test")
