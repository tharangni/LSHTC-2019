import argparse
import sl_pred
import ml_pred

def parse_pred_args():
    '''Parse input options

    Returns:
        ArgumentParser: object containing parsed arguments.

    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, help="Test dataset path")
    parser.add_argument("-f", "--features", required=True, help="Number of features", type=int)
    parser.add_argument("-t", "--hierarchy", required=True, help="Hierarchy path")
    parser.add_argument("-m", "--model_dir", required=True, help="Model directory path")
    parser.add_argument("-u", "--multi", help="Hierarchical Multi-Label", action="store_true")
    parser.add_argument("-p", "--pred_path", required=True, help="Predictions output path")


    args = parser.parse_args()
    return args


def main():
    args = parse_pred_args()
    if args.multi:
        ml_pred.main(args)
    else:
        sl_pred.main(args)

main()

