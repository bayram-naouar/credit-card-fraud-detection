from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import argparse

import preprocess
import model
from config import DATA_RAW, DATA_PROCESSED_DIR

# Parse command line arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Fraud Detection Model")
    # Main module argument group
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--isolation-forest", action="store_true", help="Use Isolation Forest model", default=False)
    model_group.add_argument("--one-class-svm", action="store_true", help="Use Isolation Forest model", default=False)

    # Optional argument group
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning", default=False)
    parser.add_argument("--save", action="store_true", help="Save model", default=False)

    return parser

# Main
def main():
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Preprocess
    preprocess.preprocess()

    # Model
    if args.isolation_forest:
        model_class = IsolationForest
    elif args.one_class_svm:
        model_class = OneClassSVM
    else:
        raise Exception("Please specify a model to use")
    
    # Run
    model.main(model_class=model_class, tune=args.tune, save=args.save)

if __name__ == "__main__":
    main()