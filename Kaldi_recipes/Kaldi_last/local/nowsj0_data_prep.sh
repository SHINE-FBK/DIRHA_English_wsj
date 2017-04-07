#!/bin/bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems. 
# - Arnab Ghoshal, 29/05/12

# Modified from the script for CHiME3 baseline
# Shinji Watanabe 02/13/2015
# Seongjun Hahm 05/05/2015

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

if [ $# -eq 1 ]; then		# default
  echo "We use the provided LMs for the experiment."
  chime3_data=$1
  cp $chime3_data/data/models/lms/lm_nowsj.o3g.kn.gz $lmdir || exit 1;
  cp $chime3_data/data/models/lms/wordlist $lmdir || exit 1;
  cp $chime3_data/data/models/lms/lm_nowsj.o4g.kn.gz $lmdir		#optional
elif [ $# -eq 2 ]; then		# LM Training routine using WSJ text
  echo "We will train LMs using WSJ text."
  np_data=$2
  for x in 87 88 89; do
    if [ ! -d $np_data/$x ]; then
      echo "$x directory is expected to exist in $np_data directory."
      exit 1;
    fi
  done
  . ./path.sh # Needed for KALDI_ROOT
  export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin

  if [ -z `which ngram-count` ]; then
    if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
      sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64 
    else
      sdir=$KALDI_ROOT/tools/srilm/bin/i686
    fi
    if [ -f $sdir/ngram-count ]; then
      echo Using SRILM tools from $sdir
      export PATH=$PATH:$sdir
    else
      echo You appear to not have SRILM tools installed, either on your path,
      echo or installed in $sdir.  See tools/install_srilm.sh for installation
      echo instructions.
      exit 1
    fi
  else
    if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
      sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64 
    else
      sdir=$KALDI_ROOT/tools/srilm/bin/i686
    fi
    export PATH=$PATH:$sdir
  fi

  set -o errexit
  export LC_ALL=C 
  cleantext=$lmdir/text.no_oov

  text=$lmdir/nowsj.txt

  # preparing text for language model training
  for x in 87 88 89; do
    for i in `find $np_data/$x -iname '*.z'`; do
      zcat $i | sed 's/\(.*\)/\U\1/;s/^<P/<p/g;s/<\/P>/<\/p>/g;s/^<S/<s/g;s/<\/S>/<\/s>/g;' | sed '/^</d' 
    done
  done > $text 

  lexicon=data/local/dict/lexicon.txt 

  # We found 10k had the best results. You can try other numbers to select the word list based on the frequency.
  heldout_sent=10000	
  cat $text | gzip -c > $lmdir/train.all.gz
  cat $text | gzip -c > $lmdir/train.gz
  cut -d' ' -f2- data/local/data/dt05_real_noisy.txt  > $lmdir/heldout

  # 10k wordlist based on word frequency
  cat data/local/lm/nowsj.txt | tr ' ' '\012' | grep "^[A-Z]" | sort | uniq -c | sort -rn | head -n 10000 | awk '{print $2}' | sort > $lmdir/wordlist

  LM=nowsj.o3g.kn
  # Trigram language model
  ngram-count -text $lmdir/train.gz -order 3 -limit-vocab -vocab $lmdir/wordlist \
    -unk -map-unk "<UNK>" -kndiscount -interpolate -lm $lmdir/${LM}.gz
  echo "PPL for NOWSJ trigram LM:"
  ngram -unk -lm $lmdir/${LM}.gz -ppl $lmdir/heldout
  ngram -unk -lm $lmdir/${LM}.gz -ppl $lmdir/heldout -debug 2 >& $lmdir/3gram.ppl2
  #file data/local/lm/heldout: 1640 sentences, 27119 words, 170 OOVs
  #0 zeroprobs, logprob= -50683.1 ppl= 59.2678 ppl1= 75.9811

  prune-lm --threshold=1e-7 $lmdir/${LM}.gz $lmdir/lm_${LM} || exit 1;
  gzip -f $lmdir/lm_${LM} || exit 1;

  LM=nowsj.o4g.kn
  # 4gram language model
  ngram-count -text $lmdir/train.gz -order 4 -limit-vocab -vocab $lmdir/wordlist \
    -unk -map-unk "<UNK>" -kndiscount -interpolate -lm $lmdir/${LM}.gz
  echo "PPL for NOWSJ 4gram LM:"
  ngram -unk -lm $lmdir/${LM}.gz -ppl $lmdir/heldout
  ngram -unk -lm $lmdir/${LM}.gz -ppl $lmdir/heldout -debug 2 >& $lmdir/4gram.ppl2
  #file data/local/lm/heldout: 1640 sentences, 27119 words, 170 OOVs
  #0 zeroprobs, logprob= -54333.4 ppl= 79.5243 ppl1= 103.79

  prune-lm --threshold=1e-7 $lmdir/${LM}.gz $lmdir/lm_${LM} || exit 1;
  gzip -f $lmdir/lm_${LM} || exit 1; 
else	# usage when there is no or greater than 3 arguments.
  printf "\nUSAGE: %s <CHiME3 root directory> <Optional:WSJ text data directory for LM Training>\n\n" `basename $0`
  echo "Please specifies a CHiME3 root directory"
  echo "If you use kaldi scripts distributed in the CHiME3 data,"
  echo "It would be `pwd`/../.."
  exit 1;
fi

echo "Data preparation succeeded"
