import argparse
import numpy as np
import torch


parser = argparse.ArgumentParser()
# basic parameters
parser.add_argument('--seed', type=int, default=4171)
parser.add_argument('--no_cuda', action='store_true', default=False)
# environment parameters
parser.add_argument('--state_dim', type=int, default=4)
parser.add_argument('--buffer_size', type=int, default=[32, 64, 128, 256])
parser.add_argument('--hyper_prd', type=int, default=256)
parser.add_argument('--slot_num', type=int, default=2048)
parser.add_argument('--flow_len', type=int, default=[1, 1])
parser.add_argument('--flow_prd', type=int, default=[2, 4, 8, 16, 32, 64, 128, 256])
parser.add_argument('--flow_delay', type=int, default=[1024, 2048])
# network parameters
parser.add_argument('--gcn_layer_dim', type=int, default=[4, 16, 64])
parser.add_argument('--q_layer_dim', type=int, default=[64, 16, 4, 1])
# training parameters
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=0.999)
parser.add_argument('--lr_decay_step', type=float, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--capacity', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta', type=float, default=1e-6)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--exploration_end_episode', type=int, default=30)
parser.add_argument('--epsilon', type=float, default=0.9)
parser.add_argument('--epsilon_decay', type=float, default=0.85)
parser.add_argument('--epsilon_decay_step', type=float, default=50)
parser.add_argument('--update_target_q_step', type=int, default=50)
parser.add_argument('--save_record_step', type=int, default=100)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)