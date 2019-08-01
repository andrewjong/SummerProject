import argparse
import generate_data as gd

parser = argparse.ArgumentParser()
parser.add_argument("--filename", default="fairnli")
parser.add_argument("--ratio", type=float, default=1, help="Difficulty. 1.0=easiest, 0=hardest")
parser.add_argument("--train_size", type=int, default=500000)
parser.add_argument("--val_size", type=int, default=10000)
parser.add_argument("--test_size", type=int, default=10000)

args = parser.parse_args()


if __name__ == '__main__':
    gd.create_corpus(args.filename, args.ratio, args.train_size, args.val_size, args.test_size)
    print("Done!")


