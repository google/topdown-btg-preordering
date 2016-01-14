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

// This program is used to preorder given sentences.

#include <fstream>
#include "misc.h"
#include "tdbtg_preorderer.h"

int main(int argc, char **argv) {
  std::string input_model = "";
  std::string input_annot = "/dev/stdin";
  std::string output_result = "/dev/stdout";
  std::string output_format = "order";
  int beam_width = 20;
  int rank_in_nbest = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-input_model") == 0 && i + 1 < argc) {
      input_model = argv[++i];
    } else if (strcmp(argv[i], "-input_annot") == 0 && i + 1 < argc) {
      input_annot = argv[++i];
    } else if (strcmp(argv[i], "-output_result") == 0 && i + 1 < argc) {
      output_result = argv[++i];
    } else if (strcmp(argv[i], "-output_format") == 0 && i + 1 < argc) {
      output_format = argv[++i];
    } else if (strcmp(argv[i], "-beam_width") == 0 && i + 1 < argc) {
      beam_width = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-rank_in_nbest") == 0 && i + 1 < argc) {
      rank_in_nbest = atoi(argv[++i]);
    } else {
      std::cerr << "usage: " << argv[0] << " [options]...\n";
      std::cerr << "  -input_model <string> : Input model file. ["
                << input_model << "]\n";
      std::cerr << "  -input_annot <string> : Input annotation file. ["
                << input_annot << "]\n";
      std::cerr << "  -output_result <string> : Output model file. ["
                << output_result << "]\n";
      std::cerr << "  -output_format <string> : Output file format: "
                << "{order, text, tree}. [" << output_format << "]\n";
      std::cerr << "  -beam_width <integer> : "
                << "Number of candidates for beam search. ["
                << beam_width << "]\n";
      std::cerr << "  -rank_in_nbest <integer> : "
                << "Rank in the n-best results to be output. ["
                << rank_in_nbest << "]\n";
      return 1;
    }
  }
  CHECK(!input_model.empty(), "Flag -input_model is empty");
  CHECK(!input_annot.empty(), "Flag -input_annot is empty");
  CHECK(!output_result.empty(), "Flag -output_result is empty");
  CHECK(output_format == "order" ||
        output_format == "text" ||
        output_format == "tree",
        "Flag --output_format should be one of {order, text, tree}");
  CHECK(!(output_format == "tree" && rank_in_nbest != 0),
        "-rank_in_nbest cannot be specified for the tree output format.");

  // Read the model data.
  TDBTGPreorderer tdbtg_preorderer;
  CHECK(beam_width >= 1, "Invalid beam width");
  tdbtg_preorderer.Configure("beam_width", std::to_string(beam_width));
  std::ifstream input_model_file(input_model);
  tdbtg_preorderer.ReadModel(&input_model_file);
  input_model_file.close();

  // Read the annotated data.
  std::vector<TDBTGPreorderer::Sentence> sentences;
  std::ifstream input_annot_file(input_annot);
  TDBTGPreorderer::ReadSentences(&input_annot_file, &sentences);
  input_annot_file.close();

  std::ofstream output_file(output_result);
  std::string result;
  for (const auto &sentence : sentences) {
    if (output_format == "tree") {
      tdbtg_preorderer.Bracket(sentence, &result);
      result.append("\n");
    } else {
      TDBTGPreorderer::Order order;
      tdbtg_preorderer.NbestPreorder(sentence, rank_in_nbest, &order);

      result.clear();
      for (unsigned int i = 0; i < order.size(); i++) {
        if (i != 0) {
          result.append(" ");
        }
        if (output_format == "text") {
          result.append(sentence[order[i]][0]);
        } else {
          result.append(std::to_string(order[i]));
        }
      }
      result.append("\n");
    }
    output_file << result;
  }
  output_file.close();

  return 0;
}
