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

  # test_dev (sim)
  dir=$data_fmllr/sim_dev
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_simdev_$1 \
     $dir data/sim_dev $gmmdir $dir/log $dir/data || exit 1
  
  # test_eval (sim)
  dir=$data_fmllr/sim_test
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_simtest_$1 \
     $dir data/sim_test $gmmdir $dir/log $dir/data || exit 1


  # test_dev (sim-only_rev)
  dir=$data_fmllr/sim_dev_onlyrev
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_simdev_$1\_onlyrev \
     $dir data/sim_dev_onlyrev $gmmdir $dir/log $dir/data || exit 1
  
  # test_eval (sim-only_rev)
  dir=$data_fmllr/sim_test_onlyrev
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_simtest_$1\_onlyrev \
     $dir data/sim_test_onlyrev $gmmdir $dir/log $dir/data || exit 1



  # test_dev (real)
  dir=$data_fmllr/real_dev
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_realdev_$1 \
     $dir data/real_dev $gmmdir $dir/log $dir/data || exit 1
  
  # test_eval (real)
  dir=$data_fmllr/real_test
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_realtest_$1 \
     $dir data/real_test $gmmdir $dir/log $dir/data || exit 1


  # test_dev (real-only rev)
  dir=$data_fmllr/real_dev_onlyrev
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_realdev_$1\_onlyrev \
     $dir data/real_dev_onlyrev $gmmdir $dir/log $dir/data || exit 1
  
  # test_eval (real-only rev)
  dir=$data_fmllr/real_test_onlyrev
  steps/nnet/make_fmllr_feats.sh --nj 6 --transform-dir $gmmdir/decode_realtest_$1\_onlyrev \
     $dir data/real_test_onlyrev $gmmdir $dir/log $dir/data || exit 1



  
  # train
  dir=$data_fmllr/tr05_cont
  steps/nnet/make_fmllr_feats.sh --nj 10 --transform-dir ${gmmdir}_ali \
     $dir data/tr05_cont $gmmdir $dir/log $dir/data || exit 1

  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1

fi

#if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
#  dir=exp/dnn_pretrain-dbn_SCW_`echo $2 | tr " " "_"`
#  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
#  $cuda_cmd $dir/log/pretrain_dbn.log \
#    local/pretrain_dbn_splice.sh --rbm-iter 3 --splice "$2" $data_fmllr/tr05_cont $dir || exit 1;
    
#fi


if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn_pretrain-dbn_dnn_rnd_SCW_`echo $2 | tr " " "_"`
  ali=${gmmdir}_ali
  feature_transform=exp/dnn_pretrain-dbn_SCW_`echo $2 | tr " " "_"`/final.feature_transform
  #dbn=exp/dnn_pretrain-dbn_SCW_`echo $2 | tr " " "_"`/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
  local/train_splice.sh --feature-transform $feature_transform --hid-layers 6 --hid_dim 2048 --learn-rate 0.008 --splice "$2" \
    $data_fmllr/tr05_cont_tr90 $data_fmllr/tr05_cont_cv10 data/lang $ali $ali $dir || exit 1;
  

  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/sim_dev $dir/decode_simdev_rnd_$1_SCW_`echo $2 | tr " " "_"` || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/sim_test $dir/decode_simtest_rnd_$1_SCW_`echo $2 | tr " " "_"` || exit 1;

  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/sim_dev_onlyrev $dir/decode_simdev_rnd_onlyrev_$1_SCW_`echo $2 | tr " " "_"` || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/sim_test_onlyrev $dir/decode_simtest_rnd_onlyrev_$1_SCW_`echo $2 | tr " " "_"` || exit 1;


  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/real_dev $dir/decode_realdev_rnd_$1_SCW_`echo $2 | tr " " "_"` || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/real_test $dir/decode_realtest_rnd_$1_SCW_`echo $2 | tr " " "_"` || exit 1;

  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/real_dev_onlyrev $dir/decode_realdev_rnd_onlyrev_$1_SCW_`echo $2 | tr " " "_"` || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph_tgpr_5k $data_fmllr/real_test_onlyrev $dir/decode_realtest_rnd_onlyrev$1_SCW_`echo $2 | tr " " "_"` || exit 1;



fi

exit
# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/dnn_pretrain-dbn_dnn_smbr
srcdir=exp/dnn_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 100 --cmd "$train_cmd" \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 100 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1; do
    steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_bd_tgpr $data_fmllr/test_dev93 $dir/decode_bd_tgpr_dev93_it${ITER} || exit 1;
    steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_bd_tgpr $data_fmllr/test_eval92 $dir/decode_bd_tgpr_eval92_it${ITER} || exit 1;
  done 
fi

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/dnn_pretrain-dbn_dnn_smbr_i1lats
srcdir=exp/dnn_pretrain-dbn_dnn_smbr
acwt=0.1

if [ $stage -le 5 ]; then
  # Generate lattices and alignments:
  steps/nnet/align.sh --nj 100 --cmd "$train_cmd" \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 100 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 6 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/train_si284 data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_bd_tgpr $data_fmllr/test_dev93 $dir/decode_bd_tgpr_dev93_iter${ITER} || exit 1;
    steps/nnet/decode.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph_bd_tgpr $data_fmllr/test_eval92 $dir/decode_bd_tgpr_eval92_iter${ITER} || exit 1;
  done 
fi

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
