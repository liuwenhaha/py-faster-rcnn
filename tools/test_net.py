#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--rpn', dest='rpn',
                        help='eval recall', action='store_true')                        

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def eval_recall(net, imdb):
    res = \
        imdb.evaluate_recall(net=net)
    ar=res["ar"]
    gt_overlaps=res["gt_overlaps"]
    recalls=res["recalls"]
    thresholds=res["thresholds"]

    print 'Method: {}'.format("RPN")
    print 'AverageRec: {}'.format(ar)

    def recall_at(t):
        ind = np.where(thresholds > t - 1e-5)[0][0]
        assert np.isclose(thresholds[ind], t)
        return recalls[ind]

    print 'Recall@0.5: {}'.format(recall_at(0.5))
    print 'Recall@0.6: {}'.format(recall_at(0.6))
    print 'Recall@0.7: {}'.format(recall_at(0.7))
    print 'Recall@0.8: {}'.format(recall_at(0.8))
    print 'Recall@0.9: {}'.format(recall_at(0.9))
    # print again for easy spreadsheet copying
    print '{:.3f}'.format(ar)
    print '{:.3f}'.format(recall_at(0.5))
    print '{:.3f}'.format(recall_at(0.6))
    print '{:.3f}'.format(recall_at(0.7))
    print '{:.3f}'.format(recall_at(0.8))
    print '{:.3f}'.format(recall_at(0.9))

    
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    
    if args.rpn:
        eval_recall(net, imdb)
    else:
        test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis, modelname=args.caffemodel, imdbname=args.imdb_name)

