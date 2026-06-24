from pathlib import Path
import kagglehub
import argparse

"""
Downlaod des WHO ATC/DDD
-> darin befinden sich die Namen der Medikamente und die daily dosage angabe
-> Link: https://www.kaggle.com/datasets/remulusbi/who-atcddd
-> Ausführen: python meaningful_modification/scripts/medical_data_download.py --outdir meaningful_modification/data/medical_information
"""

def parse_arguments():
    """
    Parses the given arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="remulusbi/who-atcddd")
    parser.add_argument("--outdir", default="./meaningful_modification/data/medical_information")

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    
    print("Loading data...")

    dataset = args.dataset
    outdir = Path(args.outdir) if args.outdir else Path(__file__).parent
    outdir.mkdir(exist_ok=True)

    path = kagglehub.dataset_download(dataset, output_dir=outdir)

    print("Path to who-atcddd dataset", path)



if __name__ == "__main__":
    main()