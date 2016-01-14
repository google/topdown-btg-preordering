// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This program is used to learn model parameters.

#include <fstream>
#include <string>
#include "misc.h"
#include "tdbtg_preorderer.h"

int main(int argc, char **argv) {
  std::string input_annot = "";
  std::string input_align = "";
  std::string output_model = "";
  int beam_width = 20;
  int training_iterations = 20;
  int min_updates = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-input_annot") == 0 && i + 1 < argc) {
      input_annot = argv[++i];
    } else if (strcmp(argv[i], "-input_align") == 0 && i + 1 < argc) {
      input_align = argv[++i];
    } else if (strcmp(argv[i], "-output_model") == 0 && i + 1 < argc) {
      output_model = argv[++i];
    } else if (strcmp(argv[i], "-beam_width") == 0 && i + 1 < argc) {
      beam_width = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-training_iterations") == 0 && i + 1 < argc) {
      training_iterations = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-min_updates") == 0 && i + 1 < argc) {
      min_updates = atoi(argv[++i]);
    } else {
      std::cerr << "usage: " << argv[0] << " [options]...\n";
      std::cerr << "  -input_annot <string> : Input annotation file. ["
                << input_annot << "]\n";
      std::cerr << "  -input_align <string> : Input alignment file. ["
                << input_align << "]\n";
      std::cerr << "  -output_model <string> : Output model file. ["
                << output_model << "]\n";
      std::cerr << "  -beam_width <integer> : "
                << "Number of candidates for beam search. ["
                << beam_width << "]\n";
      std::cerr << "  -training_iterations <integer> : "
                << "Number of training iterations of Perceptron. ["
                << training_iterations << "]\n";
      std::cerr << "  -min_updates <integer> : "
                << "Minimum updates for features not to be dropped. ["
                << min_updates << "]\n";
      return 1;
    }
  }
  CHECK(!input_annot.empty(), "Flag -input_annot is empty");
  CHECK(!input_align.empty(), "Flag -input_align is empty");
  CHECK(!output_model.empty(), "Flag -output_model is empty");

  // Set training parameters.
  TDBTGPreorderer tdbtg_preorderer;
  CHECK(beam_width >= 1, "Invalid beam width: " + std::to_string(beam_width));
  tdbtg_preorderer.Configure("beam_width", std::to_string(beam_width));
  CHECK(training_iterations >= 1, "Invalid number of training iterations: " +
        std::to_string(training_iterations));
  tdbtg_preorderer.Configure("training_iterations",
                             std::to_string(training_iterations));
  tdbtg_preorderer.Configure("min_updates", std::to_string(min_updates));

  // Read the annotation data.
  std::vector<TDBTGPreorderer::Sentence> sentences;
  std::ifstream input_annot_file(input_annot);
  TDBTGPreorderer::ReadSentences(&input_annot_file, &sentences);
  input_annot_file.close();

  // Read the alignment data.
  std::vector<TDBTGPreorderer::Alignment> alignments;
  std::ifstream input_align_file(input_align);
  TDBTGPreorderer::ReadAlignments(&input_align_file, &alignments);
  input_align_file.close();

  // Train the model.
  tdbtg_preorderer.Train(sentences, alignments);

  // Write the trained model to the file.
  std::ofstream output_model_file(output_model);
  tdbtg_preorderer.WriteModel(&output_model_file);
  output_model_file.close();

  return 0;
}
