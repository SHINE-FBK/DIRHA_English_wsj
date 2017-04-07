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

suffix=$1

dev=data-fbank/test_$suffix
train=data-fbank/train_$suffix

dev_original=data/sim_dev_onlyrev
train_original=data/tr05_cont

gmm=exp/tri4

stage=0
. utils/parse_options.sh || exit 1;

# Make the FBANK features
if [ $stage -le 0 ]; then
  # Dev set
  mkdir -p $dev && cp $dev_original/* $dev
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank_htkfalse.conf" \
     $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
  # Training set
  mkdir -p $train && cp $train_original/* $train
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" --fbank-config "conf/fbank_htkfalse.conf" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=exp/dnn4c_pretrain-dbn_$suffix
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh \
      --apply-cmvn true --norm-vars true --delta-order 2 --splice 5 \
      --hid-dim 2048 --rbm-iter 20 $train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4c_pretrain-dbn_$suffix\_dnn
  ali=${gmm}_ali
  feature_transform=exp/dnn4c_pretrain-dbn_$suffix/final.feature_transform
  dbn=exp/dnn4c_pretrain-dbn_$suffix/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmm/graph_tgpr_5k/ $dev $dir/decode || exit 1;
#  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
#    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

exit


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
dir=exp/dnn4c_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4c_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    $train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 6 iterations of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4 5 6; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmm/graph $dev $dir/decode_it${ITER} || exit 1
  done 
fi

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
