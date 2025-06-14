import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]
sys.path.append(BASE_DIR)

from utils.partseg_shapenet_util import *

if __name__ == "__main__":
    args = parse_args()

    # Encoder
    args.embed = [args.initial_dim, args.embed_dim, args.embed, args.alpha_beta]
    args.dim_ratio = list(map(lambda x:int(x), args.dim_ratio.split('-'))) # 
    args.num_blocks1 = list(map(lambda x:int(x), args.num_blocks1.split('-')))
    args.transfer_mode = list(map(lambda x:str(x), args.transfer_mode.split('-')))
    args.block1_mode = list(map(lambda x:str(x), args.block1_mode.split('-')))
    args.num_blocks2 = list(map(lambda x:int(x), args.num_blocks2.split('-')))
    args.block2_mode = list(map(lambda x:str(x), args.block2_mode.split('-')))
    args.k_neighbors = list(map(lambda x:int(x), args.k_neighbors.split('-')))
    args.sampling_mode = list(map(lambda x:str(x), args.sampling_mode.split('-')))
    args.sampling_ratio = list(map(lambda x:int(x), args.sampling_ratio.split('-')))

    # Decoder
    args.de_dims = list(map(lambda x:int(x), args.de_dims.split('-')))
    args.de_blocks = list(map(lambda x:int(x), args.de_blocks.split('-')))

    args.de_fp_fuse = list(map(lambda x:str(x), args.de_fp_fuse.split('-')))
    args.de_fp_block = list(map(lambda x:str(x), args.de_fp_block.split('-')))

    args.resume = True if args.resume=='yes' else False

    #if args.manual_seed is None:
    #    args.manual_seed = np.random.randint(1, 10000)

    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        args.msg = time_str + '-' + str(args.manual_seed)
    else:
        args.msg = args.msg + "-" + str(args.manual_seed)

    # args.msg = args.model # +"_"+args.msg

    init(args)

    if not args.eval:
        io = IOStream(f'checkpoints/partseg_shapenet/' + args.msg + '/%s_train.log' % (args.msg))
    else:
        io = IOStream(f'checkpoints/partseg_shapenet/' + args.msg + '/%s_test.log' % (args.msg))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)




