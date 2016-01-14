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

// Top-Down BTG-based preorderer.

#ifndef TDBTG_PREORDERER_H_
#define TDBTG_PREORDERER_H_

#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include "mini_hash_map.h"

class TDBTGPreorderer {
 public:
  static constexpr int kNumFeatures = 3;  // Number of features for each token.

  typedef std::vector<std::string> Token;

  typedef std::vector<Token> Sentence;

  // The index of the original token for each token in pre-ordered sentence.
  typedef std::vector<int> Order;

  // The indices of the target tokens aligned to each source token.
  typedef std::vector<std::set<int> > Alignment;

  // Read annotated data.
  static void ReadSentences(std::ifstream *file,
                            std::vector<Sentence> *sentences);

  // Read alignment data.
  static void ReadAlignments(std::ifstream *file,
                             std::vector<Alignment> *alignments);

  TDBTGPreorderer();

  ~TDBTGPreorderer();

  // Set parameters for the instance.
  // The following keys and values are accepted:
  //   - key="beam_width", value=(integer greater than 0)
  //     Set the number of candidates considered in beam search of BTG parsing.
  //   - key="training_iterations", value=(integer greater than 0)
  //     Set the number of training iterations of online Perceptron.
  //   - key="min_updates", value=(integer)
  //     Set the minimum number of updates that features are not dropped.
  void Configure(const std::string &key, const std::string &value);

  void ReadModel(std::ifstream *file);

  void WriteModel(std::ofstream *file) const;

  // Pre-order the given sentence, and return the best result.
  void Preorder(const Sentence &sentence, Order *order) const;

  // Preorder the given sentence, and return the (<rank_in_nbest> + 1)-th best
  // result in the n-best results.
  void NbestPreorder(const Sentence &sentence, int rank_in_nbest,
                     Order *order) const;

  // Preorder the given sentence, and return the BTG-bracketed sentence.
  void Bracket(const Sentence &sentence, std::string *bracket) const;

  // Train model parameters using the given sentences and word alignments.
  void Train(const std::vector<Sentence> &sentences,
             const std::vector<Alignment> &alignments);

 private:
  // Identity hash function.
  struct Hash {
    std::size_t operator()(uint64_t x) const {
      return x;
    }
  };

  // Feature set of a sentence.
  struct FeatureSet {
    size_t length;  // Sentence length.
    std::vector<std::vector<uint64_t> > sentence_fp;  // Fingerprints of tokens.
  };

  // Word order constraints.
  typedef std::vector<int> Constraint;

  // Parser action represented with pivot and label, which means to split a span
  // [bgn, end) into [bgn, pivot) and [pivot, end) with label.
  typedef std::pair<int, bool> ParserAction;

  // Span that is covered by a subtree.
  struct ParserSpan {
    int bgn;  // Beginning position of the span.
    int end;  // Ending position of the span.
    int action_id;  // Id of the action which made this span.

    ParserSpan(int bgn_, int end_, int action_id_)
        : bgn(bgn_), end(end_), action_id(action_id_) {
    }

    bool operator==(const struct ParserSpan &x) const {
      return (x.bgn == bgn && x.end == end && x.action_id == action_id);
    }
  };

  // State of the incremental parser.
  struct ParserState {
    double score;  // Accumulated score.
    bool valid;  // Whether the tree constructed so far satisfies constraints.
    std::vector<ParserSpan> stack;  // Stack of open spans.
    std::vector<ParserAction> actions;  // History of actions.

    // Make an initial state for the sentences with <len> tokens.
    explicit ParserState(int len) : score(0.0), valid(true) {
      if (len >= 2) {
        stack.emplace_back(0, len, -1);
      }
    }

    // Make a new state by applying <action> to the old state.
    ParserState(const ParserState &state, const ParserAction &action,
                double new_score, bool new_valid)
        : stack(state.stack), actions(state.actions) {
      Advance(action, new_score, new_valid);
    }

    // Change the state by applying the specified action.
    void Advance(const ParserAction &action, double new_score, bool new_valid) {
      score = new_score;
      valid = new_valid;
      const ParserSpan span = stack.back();
      stack.pop_back();
      if (action.first - span.bgn >= 2) {
        stack.emplace_back(span.bgn, action.first, actions.size());
      }
      if (span.end - action.first >= 2) {
        stack.emplace_back(action.first, span.end, actions.size());
      }
      actions.push_back(action);
    }

    bool operator<(const ParserState &rhs) const {
      // Elements with smaller scores have higher priorities.
      return (score > rhs.score);
    }
  };

  typedef std::priority_queue<ParserState> Agenda;

  struct Subtree {
    int bgn;
    int end;
    int pivot;
    bool label;

    Subtree(int bgn_, int end_, int pivot_, bool label_)
        : bgn(bgn_), end(end_), pivot(pivot_), label(label_) {
    }

    bool operator==(const struct Subtree &x) const {
      return (x.bgn == bgn && x.end == end && x.pivot == pivot &&
              x.label == label);
    }
  };

  // Parse the given sentence, and obtain n-best results.
  // In training, <constraint> should be the word order constraint, and
  // <oracle_actions> returns the action sequence satisfying the constraint.
  // In testing, <constraint> should be nullptr.
  void Parse(const FeatureSet &feature_set, const Constraint *constraint,
             std::vector<std::vector<ParserAction> > *nbest_actions,
             std::vector<ParserAction> *oracle_actions) const;

  // Check the validity of <sentence>, and convert it to a feature set.
  static void ValidateSentence(const Sentence &sentence,
                               FeatureSet *feature_set);

  // Make a feature set from <sentence>.
  static void MakeFeatureSet(const Sentence &sentence, FeatureSet *feature_set);

  // Convert the action sequence to pre-ordered token indices.
  static void ActionsToOrder(const std::vector<ParserAction> &actions,
                             Order *order);

  // Convert the action sequence to a BTG-bracketed string.
  static void ActionsToTree(const Sentence &sentence,
                            const std::vector<ParserAction> &actions,
                            std::string *tree);

  // Convert the action sequence to a subtree sequence.
  static void ActionsToSubtrees(const std::vector<ParserAction> &actions,
                                std::vector<Subtree> *subtrees);

  // Check if the constraint is BTG-parsable.
  static bool BTGParsable(const Constraint &constraint);

  // Convert word alignment to word order constraint.
  static bool AlignmentToConstraint(const Alignment &alignment,
                                    Constraint *constraint);

  // Order relation for two source tokens with word alignment information.
  static bool LessOrEqualAlignment(const std::set<int> &lhs,
                                   const std::set<int> &rhs);

  // Add a new parser state to the agenda.
  void AddState(const ParserState &state, const std::vector<uint64_t> &features,
                const ParserAction &action, bool valid,
                const Constraint *constraint, Agenda *agenda,
                double *oracle_score, std::vector<ParserAction> *oracle_actions,
                int *num_valid) const;

  // Update feature weights.
  void UpdateWeights(const FeatureSet &feature_set,
                     const std::vector<ParserAction> &actions_ref,
                     const std::vector<ParserAction> &actions_sys,
                     double coefficient);

  // Generate features for the given configuration.
  static void MakeFeatures(const FeatureSet &feature_set,
                           const ParserState &state, const ParserSpan &span,
                           int pivot, std::vector<uint64_t> *features);

  // Beam width in parsing.
  int beam_width_;

  // Number of training iterations of Perceptron.
  int training_iterations_;

  // Minimum updates for features not to be dropped.
  int min_updates_;

  // Weights of features.
  // The size of this vector is 2 (STR and INV).
  std::vector<MiniHashMap<uint64_t, float, Hash> > weights_;

  // Weights of features used in training.
  // The size of this vector is 2 (STR and INV).
  std::vector<MiniHashMap<uint64_t, float, Hash> > cached_weights_;

  // Number that each feature was updated.
  // The size of this vector is 2 (STR and INV).
  std::vector<MiniHashMap<uint64_t, int, Hash> > num_updates_;
};

#endif  // TDBTG_PREORDERER_H_
