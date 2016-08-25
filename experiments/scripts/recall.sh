GPU=$1
./tools/rpn_generate.py --gpu ${GPU} --def models/pascal_voc/VGG_CNN_M_1024/grpn/rpn_test.pt --net output/faster_rcnn_alt_opt/voc_2007_trainval/vgg_cnn_m_1024_rpn_stage1_iter_80000.caffemodel --cfg experiments/cfgs/faster_rcnn_alt_opt.yml --imdb voc_2007_test

./tools/eval_recall.py --imdb voc_2007_test --method rpn --rpn-file output/faster_rcnn_alt_opt/voc_2007_test/vgg_cnn_m_1024_rpn_stage1_iter_80000/vgg_cnn_m_1024_rpn_stage1_iter_80000_rpn_proposals.pkl