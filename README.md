# General Description:
This github project contains the kaldi baselines and the tools for the DIRHA English WSJ dataset.
You can download the data from the Linguistic Data Consortuim (LDC) website:

The wsj part of the DIRHA English Dataset [1,2] is a multi-microphone acoustic corpus being developed under the EC project Distant-speech Interaction for Robust Home Applications ([DIRHA](https://dirha.fbk.eu/)). The corpus is composed of both real and simulated sequences recorded  with 32 sample-synchronized microphones in a domestic environment. 
The database contains signals of different characteristics in terms of noise and reverberation making it suitable for various multi-microphone signal processing and distant speech recognition tasks. The part of the dataset currently released is composed of  6 native US speakers (3 Males, 3 Females) uttering  409 wsj sentences. 
The current repository provides the related Kaldi recipe and the tools that are necessary to generate the training material for kaldi-based distant speech recognizer.

In order to facilitate the use of this dataset for speech recognition purposes, a MATLAB script is released for data contamination of the original close-talking WSJ corpus (Data_Contamination.m).

The first part of the script (#TRAINING DATA) starts from the standard close-talk version of the wsj dataset and contaminates it with reverberation (using a set of impulse responses measured in the considered living-room ). The second part of this script (#DIRHA DATA) extracts the DIRHA wsj sentences of a given microphone (e.g., LA6, Beam_Circular_Array, L1R, LD07, Beam_Linear_Array, etc.) from the available 1-minute sequences. It normalizes the amplitude of each signal and performs a conversion of the xml label into a txt label. After running the script, the training and test databases are available in the specified output folder and are ready to be used within the Kaldi recipe.

The Kaldi baselines released with the DIRHA English dataset are similar to the s5 recipe proposed in the Kaldi release for WSJ data. A contaminated version of the original WSJ corpus is used for training, while test is performed with the DIRHA English wsj dataset. In short, the speech recognizer is based on standard MFCCs and acoustic models of increasing complexity. “Mono” is the simplest system based on 48 context-independent phones of the English language, each modeled by a three state left-to-right HMM. A set of context-dependent models are then derived. In “tri1”2.5k tied states with 15k gaussians are trained by exploiting a binary regression tree.“Tri2”is an evolution of the standard context-dependent model in which a Linear Discriminant Analysis (LDA) is applied. In both “tri3” and “tri4” models Speaker Adaptive Training (SAT) is also performed. The difference is that “tri4” is bootstrapped by the previously computed ‘‘tri3” model. The considered “DNN”, based on the Karel’s recipe, is composed of 6 hidden layers of 2048 neurons, with a context window of 19 consecutive frames (9 before and 9 after the current frame) and an initial learning rate of 0.008. The weights are initialized via RBM pre-training, while the fine tuning is performed with stochastic gradient descent optimizing cross-entropy loss function.


# How to run the recipe:

Note: This recipe should be run under linux (tested on RedHat and Ubuntu)

0) Download the DIRHA_English WSJ (for test purposes) as well as the original WSJ dataset (for training purposes) from the LDC website

1) If not already done, install KALDI (http://kaldi-asr.org/) and make sure that your KALDI installation is working. Try for instance to launch the original kaldi recipe in “egs/wsj/s5/run.sh” and check whether everything is properly working.

2) Generate Contaminated WSJ dataset.
  
   - A. Open Matlab
   - B. Open “Tools/Data_Contamination.m”
   - C. Set in “wsj_folder” the folder where you have stored the original (close-talking) WSJ database
   - D. Set in “out_folder” the folder where the generated datasets will be created
   - F. Select in “mic_sel” the reference microphone for the generated databases (see Additional_info/Floorplan or Additional_info/microphone_info.txt for the complete list)

3) Run the KALDI recipe.
  
   - A. Go in the “Kaldi_recipes” folder
   - B. In you are using Kaldi-trunk version, go to "kaldi_trunk". If you have the current github kaldi version (tested on 28 March 2017) go to "kaldi_last"
   - C. Open the file path.sh and set the path of your kaldi installation in “export KALDI_ROOT=your_path”.
   - D. Open the file “run.sh”
   - E. check parameters in run.sh and modify according to your machine:
        feats_nj=10 # number of jobs for feature extraction
        train_nj=30 # number of jobs for training
        decode_nj=6 # number of jobs for decoding (maximum 6)
   - F. Set directory of the contaminated wsj dataset previously created by the MATLAB script in "wsj0_folder" and "wsj0_contaminated_folder"
   - G. Set directory of the DIRHA dataset in “DIRHA_wsj_data” (e.g., dirha=DIRHA_English_wsj5k_released/Data/DIRHA_wsj_oracle_VAD_mic_LA6)
   - H. Run the script “./run.sh” .See the results by typing “./RESULTS”. Please note that the results may vary depending on: operating system, system architecture, version of kaldi. The performance obtained by us (we observed a std of about 0.6%) are repored below.

```
        mono-DIRHA_sim  WER=69.0%
        mono-DIRHA_real WER=72.1%

        tri1-DIRHA_sim  WER=45.4%
        tri1-DIRHA_real WER=51.8%

        tri2-DIRHA_sim  WER=39.6%
        tri2-DIRHA_real WER=46.3%

        tri3-DIRHA_sim  WER=31.5%
        tri3-DIRHA_real WER=37.4%

        tri4-DIRHA_sim  WER=30.4%
        tri4-DIRHA_real WER=37.2%

        dnn-DIRHA_sim   WER=22.8%
        dnn-DIRHA_real  WER=30.0%

```

# Common Issues:
- "awk:function gensub never defined”. The problem can be solved by typing the following command:  sudo apt-get install gawk
- make sure your ~/.bashrc contains the needed kaldi paths.

```PATH=$PATH:/home/kaldi-trunk/tools/openfst-1.3.4/bin
   PATH=$PATH:/home/kaldi-trunk/src/featbin
   PATH=$PATH:/home/kaldi-trunk/src/gmmbin
   PATH=$PATH:/home/kaldi-trunk/src/bin
   PATH=$PATH:/home/kaldi-trunk/src/nnetbin
   ```

# Cuda Experiments:
We recommend to use a CUDA-capable GPU for the DNN experiments. Before starting the experiments we suggest to do the following checks:

1. Make sure you have a cuda-capable GPU by typing “nvidia-smi”
2. Make sure you have installed the CUDA package (see nvidia website)
3. Make sure that in your .bashrc file you have the following lines :

       PATH=$PATH:$YOUR_CUDA_PATH/bin
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$YOUR_CUDA_PATH/lib64

4. Make sure you have installed kaldi with the cuda enabled (cd kaldi-trunk/src ; make clean; ./configure --cudatk-dir=$YOUR_CUDA_PATH; make depend ; make)
5. Test your GPU. 

    A. cd $YOUR_CUDA_PATH/samples/0_Simple/vectorAdd
    B. nvcc  vectorAdd.cu
    C. ./vectorAdd
    The result should be this: “Test PASSED”
    
    
# References:
If you use the DIRHA English wsj dataset or the related baselines and tools, please cite the following papers:

[1] M. Ravanelli, L. Cristoforetti, R. Gretter, M. Pellin, A. Sosi, M. Omologo, "The DIRHA-English corpus and related tasks for distant-speech recognition in domestic environments", in Proceedings of ASRU 2015.

[2] M. Ravanelli, P. Svaizer, M. Omologo, "Realistic Multi-Microphone Data Simulation for Distant Speech Recognition",in Proceedings of Interspeech 2016.




