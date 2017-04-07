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

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement method> <enhanced speech directory>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies the directory of enhanced wav files"
  exit 1;
fi

nj=10

# enhan data
enhan=$1
enhan_data=$2
LM=nowsj.o3g.kn

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d exp/tri3b_tr05_real_$enhan ]; then
  echo "error, execute local/run_gmm.sh, first"
  exit 1;
fi

# make 40-dim fbank features for enhan data
fbankdir=fbank/$enhan
mkdir -p data-fbank
for x in dt05_real_$enhan tr05_real_$enhan; do
  cp -r data/$x data-fbank
  steps/make_fbank.sh --nj $nj \
    data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
done

# get the number of jobs
nspk=`wc -l data/tr05_real_$enhan/spk2utt | awk '{print $1}'`
if [ $nj -gt $nspk ]; then
  nj2=$nspk
else
  nj2=$nj
fi

# get alignment
steps/align_fmllr.sh --nj $nj2 \
  data/tr05_real_$enhan data/lang exp/tri3b_tr05_real_$enhan exp/tri3b_tr05_real_${enhan}_ali || exit 1;
steps/align_fmllr.sh --nj 4 \
  data/dt05_real_$enhan data/lang exp/tri3b_tr05_real_$enhan exp/tri3b_tr05_real_${enhan}_ali_dt05 || exit 1;

# pre-train dnn
dir=exp/tri4a_dnn_pretrain_tr05_real_$enhan
$cuda_cmd $dir/_pretrain_dbn.log \
  steps/nnet/pretrain_dbn.sh --nn-depth 7 --rbm-iter 3 data-fbank/tr05_real_$enhan $dir

# train dnn
dir=exp/tri4a_dnn_tr05_real_$enhan
ali=exp/tri3b_tr05_real_${enhan}_ali
ali_dev=exp/tri3b_tr05_real_${enhan}_ali_dt05 
feature_transform=exp/tri4a_dnn_pretrain_tr05_real_$enhan/final.feature_transform
dbn=exp/tri4a_dnn_pretrain_tr05_real_$enhan/7.dbn
$cuda_cmd $dir/_train_nnet.log \
steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
data-fbank/tr05_real_$enhan data-fbank/dt05_real_$enhan data/lang $ali $ali_dev $dir || exit 1;

# decode enhan speech
utils/mkgraph.sh data/lang_test_${LM} $dir $dir/graph_${LM} || exit 1;
steps/nnet/decode.sh --nj 4 --num-threads 4 --acwt 0.10 --config conf/decode_dnn.config \
  $dir/graph_${LM} data-fbank/dt05_real_$enhan $dir/decode_${LM}_dt05_real_$enhan 
#wait;

# Sequence training using sMBR criterion, we do Stochastic-GD
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/tri4a_dnn_tr05_real_${enhan}_smbr
srcdir=exp/tri4a_dnn_tr05_real_${enhan}
acwt=0.1

# First we generate lattices and alignments:
# gawk musb be installed to perform awk -v FS="/" '{ print gensub(".gz","","",$NF)" gunzip -c "$0" |"; }' in
# steps/nnet/make_denlats.sh
steps/nnet/align.sh --nj $nj2 --cmd "$train_cmd" \
  data-fbank/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali
steps/nnet/make_denlats.sh --nj $nj2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  data-fbank/tr05_real_${enhan} data/lang $srcdir ${srcdir}_denlats

# Re-train the DNN by 1 iteration of sMBR
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
  data-fbank/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir

# Decode (reuse HCLG graph)
for ITER in 1; do
  steps/nnet/decode.sh --nj 4 --num-threads 4 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4a_dnn_tr05_real_${enhan}/graph_${LM} data-fbank/dt05_real_${enhan} $dir/decode_${LM}_dt05_real_${enhan}_it${ITER} 
done

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/tri4a_dnn_tr05_real_${enhan}_smbr_i1lats
srcdir=exp/tri4a_dnn_tr05_real_${enhan}_smbr
acwt=0.1

# Generate lattices and alignments:
steps/nnet/align.sh --nj $nj2 --cmd "$train_cmd" \
  data-fbank/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali
steps/nnet/make_denlats.sh --nj $nj2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  data-fbank/tr05_real_${enhan} data/lang $srcdir ${srcdir}_denlats

# Re-train the DNN by 4 iterations of sMBR
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
  data-fbank/tr05_real_${enhan} data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1

# Decode (reuse HCLG graph)
for ITER in 1 2 3 4; do
  steps/nnet/decode.sh --nj 4 --num-threads 4 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    exp/tri4a_dnn_tr05_real_${enhan}/graph_${LM} data-fbank/dt05_real_${enhan} $dir/decode_${LM}_dt05_real_${enhan}_it${ITER} &
done
wait

# decoded results of enhan speech using enhan DNN AMs
grep WER exp/tri4a_dnn_tr05_real_${enhan}*/decode_${LM}_dt05_real_${enhan}*/wer_* | sort -k 2 -n | head -n 1
