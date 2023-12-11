import argparse
import datetime
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset
from engine import test_wer
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer prior network', add_help=False)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # * Transformer
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")

    parser.add_argument('--enc_layers', default=6, type=int,
    help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--feed_dim', default=2048, type=int, choices=[2048, 3072],
                    help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='lrs3')
    parser.add_argument('--lrs3_path', default='', type=str)


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--hub_path', default='', type=str)
    parser.add_argument('--load_checkpoint', default=True, type=bool)

    # distributed training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    model = build_model(args)

    model.to(device)
    
    if args.load_checkpoint:
        print(f'Loading checkpoint weights from {args.model_path}')
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint, strict=False)
    
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    
    dataset_test = build_dataset(image_set='test', args=args)
    
    
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=True)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    
    batch_sampler_test = torch.utils.data.BatchSampler(
        sampler_test, args.batch_size_test, drop_last=False)

    data_loader_test = [DataLoader(dataset_test, batch_sampler=batch_sampler_test, 
                                collate_fn=utils.collate_fn_test, num_workers=args.num_workers)]
    
        
    start_time = time.time()
    
    print('Running evaluation')
    
    test_wer(model, data_loader_test, device)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASR/VSR Prior Network testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)