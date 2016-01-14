****************************************
Top-Down BTG-based Preordering
****************************************

This is an implementation of Top-Down Bracketing Transduction Grammar
(BTG)-based preordering which improves machine translation quality by
reordering an input sentence to have a word order in a target language.
The detailed algorithm can be found in this paper:

Tetsuji Nakagawa: Efficient Top-Down BTG Parsing for Machine Translation
Preordering, ACL-2015 (http://www.aclweb.org/anthology/P15-1021).


****************************************
1. Installing
****************************************

This software uses the CityHash library (https://github.com/google/cityhash),
and it needs to be installed beforehand.
After installing the library, run the following command:

$ make

Then, two binary files (tdbtg_preorderer_train and tdbtg_preorderer_parse) will
be generated.


****************************************
2. Example Usage
****************************************

The directory example/ contains tiny training and test data for
English-to-Japanese preordering. Training and testing can be carried out as
below:

* Training model parameters

$ ../tdbtg_preorderer_train \
-input_annot train.annot \
-input_align train.align \
-output_model train.model

* Preordering source sentences

$ ../tdbtg_preorderer_parse \
-input_model train.model \
-input_annot test.annot \
-output_result test.order

* Evaluating the result

$ ../evaluate_preordering.py test.align test.order

The result will look like this:

Number of evaluated sentences: 3
Number of skipped sentences: 0
Fuzzy Reordering Score: 0.857143
Kendall's Tau: 0.933333
Complete Match: 0.666667


****************************************
3. File Format
****************************************

This software uses two file formats for training and test data, Annot and
Align, which are used in Lader (http://www.phontron.com/lader/).

* Annot file

This file contains tokenized and annotated source sentences. Each line has a
sentence in the following format:

w_1 w_2 ... w_K\tp_1 p_2 ... p_K\tc_1 c_2 ... c_K

where w_i, p_i, and c_i are the i-th word, POS tag and word class respectively.
Each token is separated by a space, and each sequence of tokens is separated by
a tab.
Word classes can be obtained with Brown clustering or mkcls. If POS taggers are
not available, coarse-grained word classes will be able to be used instead
(Koo et al.: Simple Semi-supervised Dependency Parsing).

* Align file

This file contains word alignment information.
The i-th line of the file contains word alignment information of the i-th
sentence in the corresponding Annot file.
Below is the format of each line:

m-n ||| s_1-t_1 s_2-t_2 ... s_L-t_L

where m and n are the numbers of the tokens in the source and the target
sentences respectively, and each pair s_i-t_i means that the (s_i + 1)-th
source token and the (t_i + 1)-th target token are aligned.


****************************************
4. Training
****************************************

tdbtd_preorderer_train is the program for training model parameters.
It inputs training Annot and Align data, and outputs learned model data.

Available options:
  -input_annot <string>
    Input annotation file.
  -input_align <string>
    Input alignment file.
  -output_model <string>
    Output model file.
  -beam_width <integer>
    Number of candidates for beam search. The default value is 20.
  -training_iterations <integer>
    Number of training iterations of Perceptron. The default value is 20.
  -min_updates <integer>
    Minimum updates for features not to be dropped. The default value is 0.
    If an integer larger than 0 is specified, the model size is reduced by
    applying a technique for obtaining sparser Perceptron (Goldberg and
    Elhadad: Learning Sparser Perceptron Models).


****************************************
5. Preordering
****************************************

tdbtg_preorderer_parse is the program for reordering input sentences.
It inputs model data and Annot data, and outputs reordered result.

Available options:
  -input_model <string>
    Input model file.
  -input_annot <string>
    Input annotation file.
  -output_result <string>
    Output result file.
  -output_format <string>
    Output file format: {order, text, tree}.
    "order" means a sequence of numbers in which the i-th element is the index
    in the source-side of the i-th target-side token.
    "text" means a reordered sentence.
    "tree" means a BTG parse tree.
    The default value is "order".
  -beam_width <integer>
    Number of candidates for beam search. The default value is 20.
  -rank_in_nbest <integer>
    Rank in the n-best results to be output. The default value is 0.


****************************************
6. Evaluating
****************************************

evaluate_preordering.py is the program to evaluate preordering results.
It inputs a reordering result in the Order format and a gold standard word
alignment data in the Align format.

evaluate_preordering.py <align file> <order file>

Fuzzy Reordering Score and Kendall's Tau are output.
