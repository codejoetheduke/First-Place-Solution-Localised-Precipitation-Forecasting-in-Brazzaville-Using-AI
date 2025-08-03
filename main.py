import argparse
from train import run_training, run_inference

def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument(
        "--mode", choices=["train", "infer"], required=True,
        help="Mode: 'train' to train models, 'infer' to generate predictions"
    )
    parser.add_argument(
        "--output", type=str, default="submission/predictions.csv",
        help="Path to save predictions (default: submission/predictions.csv)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        print("🚀 Starting training...")
        run_training()
    elif args.mode == "infer":
        print("🔮 Running inference...")
        run_inference(args.output)

if __name__ == "__main__":
    main()