import argparse
import numpy as np
import torch


parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--seed', type=int, default=4171)
parser.add_argument('--no_cuda', action='store_true', default=False)
# environment parameters
parser.add_argument('--state_dim', type=int, default=4)
parser.add_argument('--node_capacity', type=int, default=[32, 64, 128, 256])
parser.add_argument('--hyper_prd', type=int, default=512)
parser.add_argument('--slot_num', type=int, default=4096)
parser.add_argument('--flow_len', type=int, default=[1, 1])
parser.add_argument('--flow_prd', type=int, default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
parser.add_argument('--flow_delay', type=int, default=[512, 1024])
# network parameters
parser.add_argument('--gcn_layer_dim', type=int, default=[4, 16, 64])
parser.add_argument('--q_layer_dim', type=int, default=[64, 16, 4, 1])
# training parameters
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--capacity', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.9)
parser.add_argument('--exploration_end_episode', type=int, default=500)
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--end_epsilon', type=float, default=0.03)
parser.add_argument('--epsilon_decay', type=float, default=0.99)
parser.add_argument('--update_target_q_step', type=int, default=200)
parser.add_argument('--save_record_step', type=int, default=100)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)