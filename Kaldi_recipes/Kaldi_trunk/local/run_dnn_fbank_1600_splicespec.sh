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

simdev=data-fbank/simdev_$suffix
simtest=data-fbank/simtest_$suffix
realdev=data-fbank/realdev_$suffix
realtest=data-fbank/realtest_$suffix

simdev_onlyrev=data-fbank/simdev_onlyrev_$suffix
simtest_onlyrev=data-fbank/simtest_onlyrev_$suffix
realdev_onlyrev=data-fbank/realdev_onlyrev_$suffix
realtest_onlyrev=data-fbank/realtest_onlyrev_$suffix


train=data-fbank/train_$suffix

simdev_original=data/sim_dev
simtest_original=data/sim_test
realdev_original=data/real_dev
realtest_original=data/real_test

simdev_original_onlyrev=data/sim_dev_onlyrev
simtest_original_onlyrev=data/sim_test_onlyrev
realdev_original_onlyrev=data/real_dev_onlyrev
realtest_original_onlyrev=data/real_test_onlyrev

train_original=data/tr05_cont

gmm=exp/tri4

stage=0
. utils/parse_options.sh || exit 1;

# Make the FBANK features
if [ $stage -le 0 ]; then
  
  # simdev
  mkdir -p $simdev && cp $simdev_original/* $simdev
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $simdev $simdev/log $simdev/data || exit 1;
  steps/compute_cmvn_stats.sh $simdev $simdev/log $simdev/data || exit 1;

  # simtest
  mkdir -p $simtest && cp $simtest_original/* $simtest
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $simtest $simtest/log $simtest/data || exit 1;
  steps/compute_cmvn_stats.sh $simtest $simtest/log $simtest/data || exit 1;

  
  # realdev 
  mkdir -p $realdev && cp $realdev_original/* $realdev
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $realdev $realdev/log $realdev/data || exit 1;
  steps/compute_cmvn_stats.sh $realdev $realdev/log $realdev/data || exit 1;

  # realtest
  mkdir -p $realtest && cp $realtest_original/* $realtest
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $realtest $realtest/log $realtest/data || exit 1;
  steps/compute_cmvn_stats.sh $realtest $realtest/log $realtest/data || exit 1;



  # simdev_onlyrev
  mkdir -p $simdev_onlyrev && cp $simdev_original_onlyrev/* $simdev_onlyrev
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $simdev_onlyrev $simdev_onlyrev/log $simdev_onlyrev/data || exit 1;
  steps/compute_cmvn_stats.sh $simdev_onlyrev $simdev_onlyrev/log $simdev_onlyrev/data || exit 1;

  # simtest_onlyrev
  mkdir -p $simtest_onlyrev && cp $simtest_original_onlyrev/* $simtest_onlyrev
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $simtest_onlyrev $simtest_onlyrev/log $simtest/data || exit 1;
  steps/compute_cmvn_stats.sh $simtest_onlyrev $simtest_onlyrev/log $simtest_onlyrev/data || exit 1;

  
  # realdev_onlyrev 
  mkdir -p $realdev_onlyrev && cp $realdev_original_onlyrev/* $realdev_onlyrev
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $realdev_onlyrev $realdev_onlyrev/log $realdev_onlyrev/data || exit 1;
  steps/compute_cmvn_stats.sh $realdev_onlyrev $realdev_onlyrev/log $realdev_onlyrev/data || exit 1;

  # realtest_onlyrev
  mkdir -p $realtest_onlyrev && cp $realtest_original_onlyrev/* $realtest_onlyrev
  steps/make_fbank_pitch.sh --nj 6 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $realtest_onlyrev $realtest_onlyrev/log $realtest/data || exit 1;
  steps/compute_cmvn_stats.sh $realtest_onlyrev $realtest_onlyrev/log $realtest_onlyrev/data || exit 1;

  

  # Training set
  mkdir -p $train && cp $train_original/* $train
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" --fbank-config "conf/fbank.conf" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=exp/dnn4c_pretrain-dbn_fbank_$1-SCW_`echo $2 | tr " " "_"`
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    local/pretrain_dbn_splice.sh \
      --apply-cmvn true --norm-vars true --delta-order 2 --splice "$2" \
      --hid-dim 1600 --rbm-iter 20 $train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4c_pretrain-dbn_$suffix\_dnn_fbank_$1-SCW_`echo $2 | tr " " "_"`
  ali=${gmm}_ali
  feature_transform=exp/dnn4c_pretrain-dbn_fbank_$1-SCW_`echo $2 | tr " " "_"`/final.feature_transform
  dbn=exp/dnn4c_pretrain-dbn_fbank_$1-SCW_`echo $2 | tr " " "_"`/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    local/train_splice.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 --splice "$2"\
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)

  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $simdev $dir/decode_simdev || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $simtest $dir/decode_simtest || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $realdev $dir/decode_realdev || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $realtest $dir/decode_realtest || exit 1;

  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $simdev_onlyrev $dir/decode_simdev_onlyrev || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $simtest_onlyrev $dir/decode_simtest_onlyrev || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $realdev_onlyrev $dir/decode_realdev_onlyrev || exit 1;
  steps/nnet/decode.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 $gmm/graph_tgpr_5k/ $realtest_onlyrev $dir/decode_realtest_onlyrev || exit 1;

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
