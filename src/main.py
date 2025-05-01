from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import argparse

import preprocess
import model

from config import AutoEncoderBuilder

# Parse command line arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Fraud Detection Model")
    # Main module argument group
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--isolation-forest", action="store_true", help="Use Isolation Forest model", default=False)
    model_group.add_argument("--one-class-svm", action="store_true", help="Use One Class SVM model", default=False)
    model_group.add_argument("--auto-encoder", action="store_true", help="Use AutoEncoder model", default=False)

    # Optional argument group
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning", default=False)
    parser.add_argument("--save", action="store_true", help="Save model", default=False)
    parser.add_argument("--plot", action="store_true", help="Output confusion matrix plot and top_10 results if --tune", default=False)

    return parser

# Main
def main():
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Preprocess
    print("Loading and preprocessing data...")
    preprocess.preprocess()

    # Model
    if args.isolation_forest:
        print("Using Isolation Forest model...")
        model_class = IsolationForest
    elif args.one_class_svm:
        print("Using One Class SVM model...")
        model_class = OneClassSVM
    elif args.auto_encoder:
        print("Using AutoEncoder model...")
        model_class = AutoEncoderBuilder
    else:
        raise Exception("Please specify a model to use")
    
    # Run
    model.main(model_class=model_class, tune=args.tune, save=args.save, plot=args.plot)

if __name__ == "__main__":
    main()