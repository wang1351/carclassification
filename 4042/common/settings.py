import argparse

#
#
def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,
                        default='CompCar_Res18',
                        help='Name of experiment')
    parser.add_argument("--log_out_dir", type=str,
                        default='logs',
                        help='log')
    parser.add_argument("--check_point_out_dir", type=str,
                        default='check_points',
                        help='checkout')
    parser.add_argument("--runargs_out_dir", type=str,
                        default='runargs',
                        help='runargs')
    parser.add_argument("--lr", type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument("--moment", type=float,
                        default=0.9,
                        help='momentum')
    parser.add_argument("--log_freq", type=int,
                        default=1,
                        help='logging frequency')
    parser.add_argument("--n_classes", type=int,
                        default=163,
                        help='number of classes')
    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help='batch size')
    parser.add_argument("--num_epoch", type=int,
                        default=100,
                        help='number of training epochs')
    return vars(parser.parse_args())
