#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=exp/tri4
data_fmllr=data-fmllr-tri4
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,

  # DIRHA sim
  dir=$data_fmllr/dirha_sim
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_dirha_sim \
     $dir data/dirha_sim $gmmdir $dir/log $dir/data || exit 1
  
  # DIRHA real
  dir=$data_fmllr/dirha_real
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_dirha_real \
     $dir data/dirha_real $gmmdir $dir/log $dir/data || exit 1
 
  # train
  dir=$data_fmllr/tr05_cont
  steps/nnet/make_fmllr_feats.sh --nj 10 --transform-dir ${gmmdir}_ali \
     $dir data/tr05_cont $gmmdir $dir/log $dir/data || exit 1

  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/dnn_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 3 $data_fmllr/tr05_cont $dir || exit 1;
fi


if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn_pretrain-dbn_dnn
  ali=${gmmdir}_ali
  feature_transform=exp/dnn_pretrain-dbn/final.feature_transform
  dbn=exp/dnn_pretrain-dbn/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008  \
    $data_fmllr/tr05_cont_tr90 $data_fmllr/tr05_cont_cv10 data/lang_nosp $ali $ali $dir || exit 1;
  

  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/dirha_sim $dir/decode_dirha_sim
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/dirha_real $dir/decode_dirha_real

fi


