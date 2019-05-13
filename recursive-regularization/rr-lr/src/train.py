import argparse
import ml_train
import sl_train

def parse_train_args():
    '''
    Parse input options

    Returns:
        ArgumentParser: object containing parsed arguments.

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required = True, help="Training dataset path")
    parser.add_argument("-f", "--features", required = False, help="Number of features", type=int, default=None)
    parser.add_argument("-t", "--hierarchy", required = True, help="Hierarchy path")
    parser.add_argument("-m", "--model_dir", required=True, help="Model output directory path")
    parser.add_argument("-c", "--cost_type", default = "lr",
            help="Cost type in [lr | trd | nca | etrd]. default = lr")
    parser.add_argument("-r", "--rho", help="Regularization parameter ( > 0). default=1", default=1,
                 type=float)
    parser.add_argument("-u", "--multi", help="Train multi-label classifier. Default is single-label classifier"
        , action="store_true")
    parser.add_argument("-i", "--imbalance", help="Include imbalance cost", action="store_true" )

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument("-n", "--nodes", help="Comma separated list of training nodes. By default trains" +
            " models for all leaf nodes.")
    # group.add_argument("-a", "--all", help="Train all leaf nodes", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_train_args()
    if args.multi:
        ml_train.main(args)
    else:
        sl_train.main(args)

main()
