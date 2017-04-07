#!/bin/bash

. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

feats_nj=10
train_nj=10
decode_nj=6 # Maximum 6 jobs!

wsj0_folder=/nfsmnt/moissan1/data/mravanelli/DIRHA_English_database/Data/DIRHA_English_wsj_released_github/Tools/Data_processed # Path of the training data (contaminated wsj)
wsj0_contaminated_folder=WSJ_contaminated_mic_LA6 

DIRHA_wsj_data=/nfsmnt/moissan1/data/mravanelli/DIRHA_English_database/Data/DIRHA_English_wsj_released_github/Tools/Data_processed/DIRHA_wsj_oracle_VAD_mic_LA6 # Path of the test data (DIRHA data)



# Data Preparation
local/wsj0_data_prep.sh $wsj0_folder $wsj0_contaminated_folder || exit 1;

local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

utils/prepare_lang.sh data/local/dict_nosp "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;

local/dirha_data_prep.sh $DIRHA_wsj_data/Sim 'dirha_sim' 

local/dirha_data_prep.sh $DIRHA_wsj_data/Real 'dirha_real' 

local/format_data.sh --lang-suffix "_nosp"  || exit 1;

local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;


echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set           "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc

for x in tr05_cont dirha_sim dirha_real; do 
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done

echo ============================================================================
echo "                     MonoPhone Training & Decoding                        "
echo ============================================================================

steps/train_mono.sh  --boost-silence 1.25 --nj $train_nj data/tr05_cont data/lang_nosp exp/mono  || exit 1;


utils/mkgraph.sh data/lang_nosp_test_tgpr_5k exp/mono exp/mono/graph_tgpr_5k || exit 1;


steps/decode.sh --nj $decode_nj exp/mono/graph_tgpr_5k data/dirha_sim exp/mono/decode_dirha_sim

steps/decode.sh --nj $decode_nj exp/mono/graph_tgpr_5k data/dirha_real exp/mono/decode_dirha_real


echo ============================================================================
echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
echo ============================================================================

steps/align_si.sh --boost-silence 1.25 --nj $train_nj data/tr05_cont data/lang_nosp exp/mono exp/mono_ali || exit 1;

steps/train_deltas.sh --boost-silence 1.25 2000 10000 data/tr05_cont data/lang_nosp exp/mono_ali exp/tri1 || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr_5k exp/tri1 exp/tri1/graph_tgpr_5k || exit 1;

steps/decode.sh --nj $decode_nj exp/tri1/graph_tgpr_5k data/dirha_sim exp/tri1/decode_dirha_sim || exit 1;

steps/decode.sh --nj $decode_nj exp/tri1/graph_tgpr_5k data/dirha_real exp/tri1/decode_dirha_real || exit 1;



echo ============================================================================
echo "                 tri2 : LDA + MLLT Training & Decoding                    "
echo ============================================================================

steps/align_si.sh --nj $train_nj data/tr05_cont data/lang_nosp exp/tri1 exp/tri1_ali || exit 1;

steps/train_lda_mllt.sh --splice-opts "--left-context=3 --right-context=3" 2500 15000 data/tr05_cont data/lang_nosp exp/tri1_ali exp/tri2 || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr_5k exp/tri2 exp/tri2/graph_tgpr_5k || exit 1;

steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri2/graph_tgpr_5k data/dirha_sim exp/tri2/decode_dirha_sim || exit 1;

steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri2/graph_tgpr_5k data/dirha_real exp/tri2/decode_dirha_real || exit 1;


echo ============================================================================
echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
echo ============================================================================

steps/align_si.sh  --nj $train_nj --use-graphs true data/tr05_cont data/lang_nosp exp/tri2 exp/tri2_ali  || exit 1;

steps/train_sat.sh 2500 15000 data/tr05_cont data/lang_nosp exp/tri2_ali exp/tri3 || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr_5k exp/tri3 exp/tri3/graph_tgpr_5k || exit 1;

steps/decode_fmllr.sh --nj $decode_nj exp/tri3/graph_tgpr_5k data/dirha_sim exp/tri3/decode_dirha_sim || exit 1;

steps/decode_fmllr.sh --nj $decode_nj exp/tri3/graph_tgpr_5k data/dirha_real exp/tri3/decode_dirha_real || exit 1;


echo ============================================================================
echo "              tri4 : LDA + MLLT + SAT Training & Decoding (2)                 "
echo ============================================================================

steps/align_fmllr.sh --nj $train_nj data/tr05_cont data/lang_nosp exp/tri3 exp/tri3_ali|| exit 1;

steps/train_sat.sh 4200 40000 data/tr05_cont data/lang_nosp exp/tri3_ali exp/tri4 || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr_5k exp/tri4 exp/tri4/graph_tgpr_5k || exit 1;


steps/decode_fmllr.sh --nj $decode_nj exp/tri4/graph_tgpr_5k data/dirha_sim exp/tri4/decode_dirha_sim || exit 1;

steps/decode_fmllr.sh --nj $decode_nj exp/tri4/graph_tgpr_5k data/dirha_real exp/tri4/decode_dirha_real || exit 1;



echo ============================================================================
echo "               DNN Hybrid Training & Decoding (Karel's recipe)            "
echo ============================================================================

steps/align_fmllr.sh --nj $train_nj data/tr05_cont data/lang_nosp exp/tri4 exp/tri4_ali || exit 1;

./local/run_dnn.sh



