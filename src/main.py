import preprocess
import model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning", default=False)
    parser.add_argument("--save", action="store_true", help="Save model", default=False)
    args = parser.parse_args()
    preprocess.preprocess(input_file="data/raw/creditcard.csv", output_dir="data/processed/")
    model.main(tune=args.tune, save=args.save)