#!/bin/bash

# Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
#                Inria (Emmanuel Vincent)
#                Mitsubishi Electric Research Labs (Shinji Watanabe)
#                University of Texas at Dallas (Seongjun Hahm)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script is made from the kaldi recipe of the 2nd CHiME Challenge Track 2
# made by Chao Weng

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <CHiME3 root directory>\n\n" `basename $0`
  echo "Please specifies a CHiME3 root directory"
  echo "If you use kaldi scripts distributed in the CHiME3 data,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

nj=10
# clean data
chime3_data=$1
LM=nowsj.o3g.kn		#This can be changed to nowsj.o4g.kn if you want to use 4gram LM
# The 10K vocab language model without verbalized pronunciations.
# This is used for 3rd CHiME challenge
# trigram would be: 10k frequently appread word list (170 OOV)

eval_flag=false # make it true when the evaluation data are released

# process for distant talking speech for real data
local/real_noisy_chime3_data_prep.sh $chime3_data || exit 1; 

local/wsj_prepare_dict.sh || exit 1;

local/nowsj0_data_prep.sh $chime3_data || exit 1;

utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

local/nowsj0_chime3_format_data.sh || exit 1;
 
# process for close talking speech for real data (will not be used)
local/real_close_chime3_data_prep.sh $chime3_data || exit 1;

# process for booth recording speech (will not be used)
local/bth_chime3_data_prep.sh $chime3_data || exit 1;

# process for distant talking speech for real data
local/real_noisy_chime3_data_prep.sh $chime3_data || exit 1; 

# Now make MFCC features for clean, close, and noisy data
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
# real data

if $eval_flag; then
  list="tr05_real_close tr05_real_noisy dt05_real_close dt05_real_noisy et05_real_close et05_real_noisy"
else
  list="tr05_real_close tr05_real_noisy dt05_real_close dt05_real_noisy"
fi 

mfccdir=mfcc
for x in $list; do
  steps/make_mfcc.sh --nj $nj \
    data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
done 


# training models for noisy real data
for train in tr05_real_noisy; do
  nspk=`wc -l data/$train/spk2utt | awk '{print $1}'`
  if [ $nj -gt $nspk ]; then
    nj2=$nspk
  else
    nj2=$nj
  fi

  steps/train_mono.sh --boost-silence 1.25 --nj $nj2 \
    data/$train data/lang exp/mono0a_$train || exit 1;  

  steps/align_si.sh --boost-silence 1.25 --nj $nj2 \
    data/$train data/lang exp/mono0a_$train exp/mono0a_ali_$train || exit 1;

  steps/train_deltas.sh --boost-silence 1.25 \
    2000 10000 data/$train data/lang exp/mono0a_ali_$train exp/tri1_$train || exit 1;

  steps/align_si.sh --nj $nj2 \
    data/$train data/lang exp/tri1_$train exp/tri1_ali_$train || exit 1;

  steps/train_lda_mllt.sh \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 data/$train data/lang exp/tri1_ali_$train exp/tri2b_$train || exit 1;

  steps/align_si.sh  --nj $nj2 \
    --use-graphs true data/$train data/lang exp/tri2b_$train exp/tri2b_ali_$train  || exit 1;

  steps/train_sat.sh \
    2500 15000 data/$train data/lang exp/tri2b_ali_$train exp/tri3b_$train || exit 1; 

  utils/mkgraph.sh data/lang_test_$LM exp/tri3b_$train exp/tri3b_$train/graph_$LM || exit 1;

  # if you want to know the result of the close talk microphone, plese try the following
  # decode close speech
  #steps/decode_fmllr.sh --nj 4 --num-threads 4 \
  #   exp/tri3b_$train/graph_tgpr_5k data/dt05_real_close exp/tri3b_$train/decode_tgpr_5k_dt05_real_close &
  # decode noisy speech
  steps/decode_fmllr.sh --nj 4 --num-threads 4 \
    exp/tri3b_$train/graph_$LM data/dt05_real_noisy exp/tri3b_$train/decode_${LM}_dt05_real_noisy &
done
wait

# get the best scores
for train in tr05_real_noisy; do
  grep WER exp/tri3b_$train/decode_${LM}_dt05_real_noisy/wer_* | sort -k 2 -n | head -n 1
done
