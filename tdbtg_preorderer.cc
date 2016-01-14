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

#include <city.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <unordered_map>
#include "misc.h"
#include "tdbtg_preorderer.h"

namespace {

inline uint64_t FeatureFingerprint(const std::string &s) {
  return CityHash64(s.data(), s.size());
}

uint64_t *FeatureFingerprintNumbers(int n) {
  uint64_t *a = new uint64_t[n];
  for (int i = 0; i < n; i++) {
    a[i] = FeatureFingerprint(std::to_string(i));
  }
  return a;
}

inline uint64_t FeatureFingerprintCat(uint64_t fp1, uint64_t fp2) {
  return Hash128to64(uint128(fp1, fp2));
}

inline uint64_t FeatureFingerprintCat(uint64_t fp1, uint64_t fp2,
                                      uint64_t fp3) {
  return FeatureFingerprintCat(FeatureFingerprintCat(fp1, fp2), fp3);
}

inline uint64_t FeatureFingerprintCat(uint64_t fp1, uint64_t fp2, uint64_t fp3,
                                      uint64_t fp4) {
  return FeatureFingerprintCat(FeatureFingerprintCat(fp1, fp2, fp3), fp4);
}

inline uint64_t FeatureFingerprintCat(uint64_t fp1, uint64_t fp2, uint64_t fp3,
                                      uint64_t fp4, uint64_t fp5) {
  return FeatureFingerprintCat(FeatureFingerprintCat(fp1, fp2, fp3, fp4), fp5);
}

}  // namespace

void TDBTGPreorderer::ReadSentences(
    std::ifstream *file, std::vector<TDBTGPreorderer::Sentence> *sentences) {
  for (std::string line; !getline(*file, line).eof(); ) {
    sentences->resize(sentences->size() + 1);
    TDBTGPreorderer::Sentence *sentence = &sentences->back();
    const std::vector<std::string> sequences = Split(line, "\t");
    for (size_t i = 0; i < sequences.size(); i++) {
      const std::vector<std::string> tokens = Split(sequences[i], " ");
      if (i == 0) {
        sentence->resize(tokens.size());
      } else {
        CHECK(tokens.size() == sentence->size(), "Invalid line: " + line);
      }
      for (size_t j = 0; j < tokens.size(); j++) {
        (*sentence)[j].push_back(tokens[j]);
      }
    }
  }
}

void TDBTGPreorderer::ReadAlignments(
    std::ifstream *file, std::vector<TDBTGPreorderer::Alignment> *alignments) {
  for (std::string line; getline(*file, line); ) {
    alignments->resize(alignments->size() + 1);
    TDBTGPreorderer::Alignment *alignment = &alignments->back();
    const std::vector<std::string> fields = Split(line, " ");
    CHECK(fields.size() >= 3, "Invalid line: " + line);
    CHECK(fields[1] == "|||", "Invalid line: " + line);
    const std::vector<std::string> nums = Split(fields[0], "-");
    CHECK(nums.size() == 2, "Invalid line: " + line);
    const int src_num = std::stoi(nums[0]);
    const int trg_num = std::stoi(nums[1]);
    alignment->resize(src_num);
    for (size_t i = 2; i < fields.size(); i++) {
      const std::vector<std::string> ids = Split(fields[i], "-");
      CHECK(ids.size() == 2, "Invalid line: " + line);
      const int src_id = std::stoi(ids[0]);
      const int trg_id = std::stoi(ids[1]);
      CHECK(src_id >= 0, "Invalid line: " + line);
      CHECK(src_id < src_num, "Invalid line: " + line);
      CHECK(trg_id >= 0, "Invalid line: " + line);
      CHECK(trg_id < trg_num, "Invalid line: " + line);
      (*alignment)[src_id].insert(trg_id);
    }
  }
}

TDBTGPreorderer::TDBTGPreorderer()
    : weights_(2, MiniHashMap<uint64_t, float, Hash>(HUGE_VALF)),
      cached_weights_(2, MiniHashMap<uint64_t, float, Hash>(HUGE_VALF)),
      num_updates_(2, MiniHashMap<uint64_t, int, Hash>(INT_MIN)) {
}

TDBTGPreorderer::~TDBTGPreorderer() {
}

void TDBTGPreorderer::Configure(const std::string &key,
                                const std::string &value) {
  if (key == "beam_width") {
    beam_width_ = std::stoi(value);
  } else if (key == "training_iterations") {
    training_iterations_ = std::stoi(value);
  } else if (key == "min_updates") {
    min_updates_ = std::stoi(value);
  } else {
    CHECK(false, "Unknown config key: " + key);
  }
}

void TDBTGPreorderer::ReadModel(std::ifstream *file) {
  int num = 0;
  for (int i = 0; i < 2; i++) {
    size_t size;
    CHECK(file->read(reinterpret_cast<char *>(&size), sizeof(size)),
          "Cannot read the model data");
    weights_[i].clear();
    weights_[i].resize(size);
    for (size_t j = 0; j < size; j++) {
      uint64_t feature;
      CHECK(file->read(reinterpret_cast<char *>(&feature),
                       sizeof(feature)).good(), "Cannot read the model data");
      float weight;
      CHECK(file->read(reinterpret_cast<char *>(&weight),
                       sizeof(weight)).good(), "Cannot read the model data");
      weights_[i][feature] = weight;
      num++;
    }
  }
  std::cerr << num << " feature weights were read." << std::endl;
}

void TDBTGPreorderer::WriteModel(std::ofstream *file) const {
  for (int i = 0; i < 2; i++) {
    const size_t size = weights_[i].size();
    CHECK(file->write(reinterpret_cast<const char *>(&size),
                      sizeof(size)).good(), "Cannot write the model data");
    for (auto it = weights_[i].begin(); it != weights_[i].end(); ++it) {
      const uint64_t feature = it->first;
      CHECK(file->write(reinterpret_cast<const char *>(&feature),
                        sizeof(feature)).good(), "Cannot write the model data");
      const float weight = it->second;
      CHECK(file->write(reinterpret_cast<const char *>(&weight),
                        sizeof(weight)).good(), "Cannot write the model data");
    }
  }
}

void TDBTGPreorderer::Preorder(const Sentence &sentence, Order *order) const {
  order->clear();
  if (sentence.empty()) {
    return;
  }

  FeatureSet feature_set;
  ValidateSentence(sentence, &feature_set);
  std::vector<std::vector<ParserAction> > nbest_actions;
  Parse(feature_set, nullptr, &nbest_actions, nullptr);

  ActionsToOrder(nbest_actions.front(), order);
}

void TDBTGPreorderer::NbestPreorder(const Sentence &sentence, int rank_in_nbest,
                                    Order *order) const {
  order->clear();
  if (sentence.empty()) {
    return;
  }

  FeatureSet feature_set;
  ValidateSentence(sentence, &feature_set);
  std::vector<std::vector<ParserAction> > nbest_actions;
  Parse(feature_set, nullptr, &nbest_actions, nullptr);

  // Remove duplicates and find the (rank_in_nbest + 1)-th result in n-best.
  int results_to_be_dropped = rank_in_nbest;
  std::set<std::vector<int>> dup;
  for (size_t i = 0; i < nbest_actions.size(); i++) {
    ActionsToOrder(nbest_actions[i], order);
    if (dup.count(*order)) continue;
    if (results_to_be_dropped == 0) {
      return;
    }
    results_to_be_dropped--;
    dup.insert(*order);
  }
  // Return the top result when the n-best results were exhausted.
  ActionsToOrder(nbest_actions.front(), order);
}

void TDBTGPreorderer::Bracket(const Sentence &sentence,
                              std::string *bracket) const {
  bracket->clear();
  if (sentence.empty()) {
    return;
  }

  FeatureSet feature_set;
  ValidateSentence(sentence, &feature_set);
  std::vector<std::vector<ParserAction> > nbest_actions;
  Parse(feature_set, nullptr, &nbest_actions, nullptr);

  ActionsToTree(sentence, nbest_actions.front(), bracket);
}

void TDBTGPreorderer::Train(const std::vector<Sentence> &sentences,
                            const std::vector<Alignment> &alignments) {
  std::cerr << "# Beam width: " << beam_width_ << std::endl;
  std::cerr << "# Training iterations: " << training_iterations_ << std::endl;
  std::cerr << "# Minimum updates: " << min_updates_ << std::endl;

  // Validity check.
  CHECK(sentences.size() == alignments.size(), "Invalid training data");
  for (size_t i = 0; i < sentences.size(); i++) {
    CHECK(sentences[i].size() >= 1, "Invalid trainig data");
    CHECK(sentences[i].size() == alignments[i].size(), "Invalid training data");
    for (size_t j = 0; j < sentences[i].size(); j++) {
      CHECK(sentences[i][j].size() == kNumFeatures, "Invalid traiing data");
    }
  }
  std::cerr << "# Number of training sentences: " << sentences.size()
            << std::endl;

  // Make word order constraints from word alignment.
  std::vector<const Sentence *> valid_sentences;
  std::vector<Constraint> constraints;
  for (size_t l = 0; l < sentences.size(); l++) {
    Constraint constraint;
    if (AlignmentToConstraint(alignments[l], &constraint)) {
      valid_sentences.push_back(&sentences[l]);
      constraints.push_back(constraint);
    }
  }
  std::cerr<< "# Number of sentences with valid constraints: "
            << valid_sentences.size() << std::endl;

  // Initialize feature weights.
  for (int i = 0; i < 2; i++) {
    weights_[i].clear();
    cached_weights_[i].clear();
    num_updates_[i].clear();
  }

  int num_updates = 0;
  const int total_updates = training_iterations_ * valid_sentences.size();

  for (int iter = 0; iter < training_iterations_; iter++) {
    std::cerr << "Iteration=" << iter << std::endl;
    int num_errors = 0;
    int num_unreachables = 0;

    for (size_t l = 0; l < valid_sentences.size(); l++) {
      if (l % 1000 == 0) {
        std::cerr << " Sentences=" << l << std::endl;
      }

      FeatureSet feature_set;
      MakeFeatureSet(*valid_sentences[l], &feature_set);

      std::vector<std::vector<ParserAction> > nbest_actions;
      std::vector<ParserAction> oracle_actions;
      std::vector<ParserAction> actions_ref;
      Parse(feature_set, &constraints[l], &nbest_actions, &actions_ref);
      const std::vector<ParserAction> &actions_sys = nbest_actions.front();

      if (actions_ref.empty()) {
        num_unreachables++;
      } else if (actions_sys.size() != valid_sentences[l]->size() - 1 ||
                 actions_sys != actions_ref) {
        num_errors++;
        const double coefficient = static_cast<double>(
            total_updates - num_updates) / total_updates;
        UpdateWeights(feature_set, actions_ref, actions_sys, coefficient);
      }
      num_updates++;
    }
    std::cerr << "Number of errors: " << num_errors
              << ", Number of unreachables: " << num_unreachables << std::endl;
    if (num_errors == 0) break;
  }

  for (int i = 0; i < 2; i++) {
    weights_[i].clear();
    for (auto it = cached_weights_[i].begin(); it != cached_weights_[i].end();
         ++it) {
      if (it->second != 0.0 && num_updates_[i][it->first] >= min_updates_) {
        weights_[i][it->first] = it->second;
      }
    }
    cached_weights_[i].clear();
    num_updates_[i].clear();
  }
}

void TDBTGPreorderer::Parse(
    const FeatureSet &feature_set, const Constraint *constraint,
    std::vector<std::vector<ParserAction> > *nbest_actions,
    std::vector<ParserAction> *oracle_actions) const {
  Agenda old_agenda;
  Agenda new_agenda;
  // Add the initial parser state to the agenda.
  old_agenda.push(ParserState(feature_set.length));

  std::vector<uint64_t> features;
  for (size_t transition = 0; transition < feature_set.length - 1;
       transition++) {
    int num_valid = 0;
    double oracle_score = -DBL_MAX;
    if (constraint != nullptr) {
      oracle_actions->clear();
    }
    // Incremental top-down parsing with beam search.
    // For each previous state stored in <old_agenda>, all the possible actions
    // are applied in order to create new states, and they are stored to
    // <new_agenda>.
    while (!old_agenda.empty()) {
      const ParserState &state = old_agenda.top();
      const ParserSpan &span = state.stack.back();

      // Pre-calculate minimum/maximum elements in the span.
      std::vector<int> lmin(span.end - span.bgn);
      std::vector<int> lmax(span.end - span.bgn);
      std::vector<int> rmin(span.end - span.bgn);
      std::vector<int> rmax(span.end - span.bgn);
      if (constraint != nullptr) {
        int cur_lmin = -1;
        int cur_lmax = -1;
        for (int i = 0; i < span.end - span.bgn; i++) {
          const int c = (*constraint)[span.bgn + i];
          if (c >= 0) {
            if (cur_lmin < 0 || c < cur_lmin) {
              cur_lmin = c;
            }
            if (cur_lmax < 0 || c > cur_lmax) {
              cur_lmax = c;
            }
          }
          lmin[i] = cur_lmin;
          lmax[i] = cur_lmax;
        }
        int cur_rmin = -1;
        int cur_rmax = -1;
        for (int i = span.end - span.bgn - 1; i >= 0; i--) {
          const int c = (*constraint)[span.bgn + i];
          if (c >= 0) {
            if (cur_rmin < 0 || c < cur_rmin) {
              cur_rmin = c;
            }
            if (cur_rmax < 0 || c > cur_rmax) {
              cur_rmax = c;
            }
          }
          rmin[i] = cur_rmin;
          rmax[i] = cur_rmax;
        }
      }

      // Check the possibility that two children [span.bgn, pivot) and
      // [pivot, span.end) have STR and INV nodes.
      for (int pivot = span.bgn + 1; pivot < span.end; pivot++) {
        bool valid = false;
        MakeFeatures(feature_set, state, span, pivot, &features);
        // Consider the tree with STR node split at <pivot>.
        if (constraint != nullptr) {
          valid = (lmax[pivot - 1 - span.bgn] < 0 ||
                   rmin[pivot - span.bgn] < 0 ||
                   lmax[pivot - 1 - span.bgn] <= rmin[pivot - span.bgn]) &&
                  state.valid;
        }
        AddState(state, features, ParserAction(pivot, false), valid, constraint,
                 &new_agenda, &oracle_score, oracle_actions, &num_valid);
        // Consider the tree with INV node split at <pivot>.
        if (constraint != nullptr) {
          valid = (rmax[pivot - span.bgn] < 0 ||
                   lmin[pivot - 1 - span.bgn] < 0 ||
                   rmax[pivot - span.bgn] <= lmin[pivot - 1 - span.bgn]) &&
                  state.valid;
        }
        AddState(state, features, ParserAction(pivot,  true), valid, constraint,
                 &new_agenda, &oracle_score, oracle_actions, &num_valid);
      }
      old_agenda.pop();
    }
    CHECK(new_agenda.size() >= 1, "No valid candidates");
    old_agenda.swap(new_agenda);

    if (constraint != nullptr && num_valid == 0) {
      // Early update.
      nbest_actions->resize(1);
      (*nbest_actions)[0] = old_agenda.top().actions;
      return;
    }
  }

  nbest_actions->resize(old_agenda.size());
  for (auto it = nbest_actions->rbegin(); it != nbest_actions->rend(); ++it) {
    const ParserState &state = old_agenda.top();
    CHECK(state.actions.size() == feature_set.length - 1,
          "Invalid length of an action sequence");
    CHECK(state.stack.size() == 0, "Stack is not empty");
    *it = state.actions;
    old_agenda.pop();
  }
}

void TDBTGPreorderer::ValidateSentence(const Sentence &sentence,
                                       FeatureSet *feature_set) {
  // Validity check.
  for (const Token &token : sentence) {
    CHECK(token.size() == kNumFeatures,
          "The format of input sentences is invalid");
  }

  // Make a feature set.
  MakeFeatureSet(sentence, feature_set);
}

void TDBTGPreorderer::MakeFeatureSet(const Sentence &sentence,
                                     FeatureSet *feature_set) {
  feature_set->length = sentence.size();

  // Convert words/pos-tags/word-clusters to fingerprints.

  feature_set->sentence_fp.resize(sentence.size());
  for (size_t i = 0; i < sentence.size(); i++) {
    for (int j = 0; j < 3; j++) {
      feature_set->sentence_fp[i].push_back(FeatureFingerprint(sentence[i][j]));
    }
  }
}

void TDBTGPreorderer::ActionsToOrder(const std::vector<ParserAction> &actions,
                                     Order *order) {
  std::vector<Subtree> subtrees;
  ActionsToSubtrees(actions, &subtrees);

  order->clear();
  for (size_t i = 0; i < actions.size() + 1; i++) {
    order->push_back(i);
  }

  // Traverse the nodes in a right-to-left bottom-up way.
  while (!subtrees.empty()) {
    const Subtree &subtree = subtrees.back();
    if (subtree.label) {
      order->insert(order->begin() + subtree.end, order->begin() + subtree.bgn,
                    order->begin() + subtree.pivot);
      order->erase(order->begin() + subtree.bgn,
                   order->begin() + subtree.pivot);
    }
    subtrees.pop_back();
  }
}

void TDBTGPreorderer::ActionsToTree(const Sentence &sentence,
                                    const std::vector<ParserAction> &actions,
                                    std::string *tree) {
  std::vector<Subtree> subtrees;
  ActionsToSubtrees(actions, &subtrees);

  // Make an escaped word sequence.
  std::vector <std::string> words;
  for (const auto &token : sentence) {
    std::string word(token[0]);
    Replace("\\", "\\\\", &word);
    Replace("[", "\\[", &word);
    Replace("]", "\\]", &word);
    Replace("<", "\\<", &word);
    Replace(">", "\\>", &word);
    words.push_back(word);
  }

  while (!subtrees.empty()) {
    const Subtree &subtree = subtrees.back();
    if (subtree.label) {
      words[subtree.bgn].insert(0, "<");
      words[subtree.end - 1].append(">");
    } else {
      words[subtree.bgn].insert(0, "[");
      words[subtree.end - 1].append("]");
    }
    subtrees.pop_back();
  }
  *tree = Join(words, " ");
}

void TDBTGPreorderer::ActionsToSubtrees(
    const std::vector<ParserAction> &actions,
    std::vector<Subtree> *subtrees) {
  ParserState state(actions.size() + 1);
  for (const ParserAction &action : actions) {
    const ParserSpan &span = state.stack.back();
    subtrees->emplace_back(span.bgn, span.end, action.first, action.second);
    state.Advance(action, 0.0, false);
  }
}

bool TDBTGPreorderer::BTGParsable(const Constraint &constraint) {
  if (constraint.empty()) return false;

  std::vector<int> lmin(constraint.size() - 1);
  std::vector<int> lmax(constraint.size() - 1);
  std::vector<int> rmin(constraint.size() - 1);
  std::vector<int> rmax(constraint.size() - 1);
  std::vector<std::pair<int, int> > stack;

  std::vector<int> positions;
  for (size_t i = 0; i < constraint.size() - 1; i++) {
    if (constraint[i] >= 0) positions.push_back(constraint[i]);
  }

  stack.push_back({0, positions.size()});
  while (!stack.empty()) {
    const auto span = stack.back();
    stack.pop_back();
    for (int i = 0; i < span.second - span.first; i++) {
      const int p = positions[span.first + i];
      if (i == 0) {
        lmin[i] = p;
        lmax[i] = p;
      } else {
        lmin[i] = std::min(lmin[i - 1], p);
        lmax[i] = std::max(lmax[i - 1], p);
      }
    }
    for (int i = span.second - span.first - 1; i >= 0; i--) {
      const int p = positions[span.first + i];
      if (i == span.second - span.first - 1) {
        rmin[i] = p;
        rmax[i] = p;
      } else {
        rmin[i] = std::min(rmin[i + 1], p);;
        rmax[i] = std::max(rmax[i + 1], p);;
      }
    }
    int split = -1;
    for (int i = 1; i < span.second - span.first; i++) {
      if (lmax[i - 1] <= rmin[i] || rmax[i] <= lmin[i - 1]) {
        split = i;
        break;
      }
    }
    if (split < 0) return false;
    const int third = span.first + split;
    if (third - span.first > 1) {
      stack.push_back({span.first, third});
    }
    if (span.second - third > 1) {
      stack.push_back({third, span.second});
    }
  }
  return true;
}

bool TDBTGPreorderer::AlignmentToConstraint(const Alignment &alignment,
                                            Constraint *constraint) {
  std::vector<std::vector<int> > sorted_indices;

  // Sort the indices of the source-side tokens according to the positions of
  // the corresponding target-side tokens.
  for (size_t i = 0; i < alignment.size(); i++) {
    if (alignment[i].empty()) continue;
    bool eq = false;
    size_t j;
    for (j = 0; j < sorted_indices.size(); j++) {
      const bool le = LessOrEqualAlignment(
          alignment[i], alignment[sorted_indices[j].front()]);
      const bool ge = LessOrEqualAlignment(
          alignment[sorted_indices[j].front()], alignment[i]);
      if (!le && !ge) return false;
      eq = (le && ge);
      if (le) break;
    }
    if (!eq) {
      sorted_indices.insert(sorted_indices.begin() + j, std::vector<int>());
    }
    sorted_indices[j].push_back(i);
  }

  constraint->assign(alignment.size(), -1);
  for (size_t i = 0; i < sorted_indices.size(); i++) {
    for (const int j : sorted_indices[i]) {
      (*constraint)[j] = i;
    }
  }
  // Push the number of target-side tokens at the end of the vector.
  constraint->push_back(sorted_indices.size());
  return BTGParsable(*constraint);
}

bool TDBTGPreorderer::LessOrEqualAlignment(const std::set<int> &lhs,
                                           const std::set<int> &rhs) {
  for (const int x : lhs) {
    if (rhs.find(x) == rhs.end()) {
      for (const int y : rhs) {
        if (x > y) return false;
      }
    }
  }
  for (const int y : rhs) {
    if (lhs.find(y) == lhs.end()) {
      for (const int x : lhs) {
        if (x > y) return false;
      }
    }
  }
  return true;
}

void TDBTGPreorderer::AddState(const ParserState &state,
                               const std::vector<uint64_t> &features,
                               const ParserAction &action, bool valid,
                               const Constraint *constraint, Agenda *agenda,
                               double *oracle_score,
                               std::vector<ParserAction> *oracle_actions,
                               int *num_valid) const {
  // Calculate the score.
  double score = state.score;
  const MiniHashMap<uint64_t, float, Hash> &weights =
      weights_[static_cast<int>(action.second)];
  for (const uint64_t feature : features) {
    const auto it = weights.find(feature);
    if (it != weights.end()) {
      score += it->second;
    }
  }

  if (valid && score > *oracle_score) {
    *oracle_score = score;
    *oracle_actions = state.actions;
    oracle_actions->push_back(action);
  }

  // Remove the candidate with the least score if the agenda size exceeds the
  // beam size.
  if (agenda->size() >= static_cast<size_t>(beam_width_)) {
    if (score <= agenda->top().score) return;
    if (agenda->top().valid) {
      (*num_valid)--;
    }
    agenda->pop();
  }

  if (valid) {
    (*num_valid)++;
  }
  agenda->emplace(state, action, score, valid);
}

// Update the feature weights using the Passive-Aggressive algorithm (PA-I).
void TDBTGPreorderer::UpdateWeights(
    const FeatureSet &feature_set,
    const std::vector<ParserAction> &actions_ref,
    const std::vector<ParserAction> &actions_sys,
    double coefficient) {
  CHECK(actions_ref.size() == actions_sys.size(),
        "Inconsistent lengths of action sequences");
  ParserState state_ref(feature_set.length);
  ParserState state_sys(feature_set.length);
  int start = actions_ref.size();
  for (size_t i = 0; i < actions_ref.size(); i++) {
    if (actions_ref[i] != actions_sys[i]) {
      start = i;
      break;
    }
    state_ref.Advance(actions_ref[i], 0.0, false);
    state_sys.Advance(actions_sys[i], 0.0, false);
  }

  std::vector<uint64_t> features;
  double loss = 0.0;
  std::vector<std::unordered_map<uint64_t, int> > features_diff(2);
  for (size_t i = start; i < actions_ref.size(); i++) {
    const ParserSpan &span_ref = state_ref.stack.back();
    const ParserAction &action_ref = actions_ref[i];
    MakeFeatures(feature_set, state_ref, span_ref, action_ref.first, &features);
    state_ref.Advance(action_ref, 0.0, false);
    const MiniHashMap<uint64_t, float, Hash> &weights_ref =
        weights_[static_cast<int>(action_ref.second)];
    std::unordered_map<uint64_t, int> &features_diff_ref =
        features_diff[static_cast<int>(action_ref.second)];
    for (const uint64_t feature : features) {
      const auto it = weights_ref.find(feature);
      if (it != weights_ref.end()) {
        loss -= it->second;
      }
      if (features_diff_ref.find(feature) == features_diff_ref.end()) {
        features_diff_ref[feature] = +1;
      } else {
        features_diff_ref[feature] += 1;
      }
    }

    const ParserSpan &span_sys = state_sys.stack.back();
    const ParserAction &action_sys = actions_sys[i];
    MakeFeatures(feature_set, state_sys, span_sys, action_sys.first, &features);
    state_sys.Advance(action_sys, 0.0, false);
    const MiniHashMap<uint64_t, float, Hash> &weights_sys =
        weights_[static_cast<int>(action_sys.second)];
    std::unordered_map<uint64_t, int> &features_diff_sys =
        features_diff[static_cast<int>(action_sys.second)];
    for (const uint64_t feature : features) {
      const auto it = weights_sys.find(feature);
      if (it != weights_sys.end()) {
        loss += it->second;
      }
      if (features_diff_sys.find(feature) == features_diff_sys.end()) {
        features_diff_sys[feature] = -1;
      } else {
        features_diff_sys[feature] -= 1;
      }
    }
  }

  // Calculate the loss.
  loss += sqrt(static_cast<double>(actions_ref.size() - start));

  // Calculate tau.
  int sqnorm = 0;
  for (int i = 0; i < 2; i++) {
    for (auto it = features_diff[i].begin(); it != features_diff[i].end();
         ++it) {
      sqnorm += it->second * it->second;
    }
  }
  const double tau = std::min(1.0, loss / sqnorm);

  for (int i = 0; i < 2; i++) {
    for (auto it = features_diff[i].begin(); it != features_diff[i].end();
         ++it) {
      weights_[i][it->first] += tau * it->second;
      cached_weights_[i][it->first] += tau * it->second * coefficient;
      num_updates_[i][it->first]++;
    }
  }
}

void TDBTGPreorderer::MakeFeatures(const FeatureSet &feature_set,
                                   const ParserState &state,
                                   const ParserSpan &span, int pivot,
                                   std::vector<uint64_t> *features) {
  static const int kMaxFeatures = 110;
  static const int kMaxNumbers = 100;
  static const uint64_t kFpEmpty = FeatureFingerprint("");
  static const uint64_t *kFpNumbers =
      FeatureFingerprintNumbers(std::max(kMaxFeatures, kMaxNumbers));
  int id = 0;

  const std::vector<std::vector<uint64_t> > &sentence_fp =
      feature_set.sentence_fp;

  features->clear();
  features->reserve(kMaxFeatures);

  const int lchild_bgn = span.bgn;
  const int lchild_end = pivot;
  const int rchild_bgn = pivot;
  const int rchild_end = span.end;
  const int sentence_size = static_cast<int>(sentence_fp.size());

  const int sn = (rchild_end - lchild_bgn >= kMaxNumbers) ?
      (kMaxNumbers - 1) : (rchild_end - lchild_bgn);
  const int ln = lchild_end - lchild_bgn;
  const int rn = rchild_end - rchild_bgn;
  const int cd = rn - ln;
  const int cl = (cd < 0) ? 0 : (cd > 0) ? 1 : 2;
  const int nln = (ln > 5) ? 5 : ln;
  const int nrn = (rn > 5) ? 5 : rn;

  // bias
  features->push_back(kFpNumbers[id++]);
  // balance of children
  features->push_back(FeatureFingerprintCat(
      kFpNumbers[id++], kFpNumbers[cl]));
  // tree size
  features->push_back(FeatureFingerprintCat(
      kFpNumbers[id++], kFpNumbers[sn]));
  // subtree sizes
  features->push_back(FeatureFingerprintCat(
      kFpNumbers[id++], kFpNumbers[nln], kFpNumbers[nrn]));

  int gp_nt;
  int gp_side;
  if (span.action_id < 0) {
    gp_nt = 2;
    gp_side = 2;
  } else {
    const ParserAction &action = state.actions[span.action_id];
    gp_nt = static_cast<int>(action.second);
    gp_side = static_cast<int>(span.bgn == action.first);
  }
  // NT
  features->push_back(FeatureFingerprintCat(
      kFpNumbers[id++], kFpNumbers[gp_nt]));
  // NT&SIDE
  features->push_back(FeatureFingerprintCat(
      kFpNumbers[id++], kFpNumbers[gp_nt], kFpNumbers[gp_side]));

  for (int i = 0; i < 3; i++) {
    // Ll-
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++],
        (lchild_bgn - 1 >= 0) ? sentence_fp[lchild_bgn - 1][i] : kFpEmpty));
    // Ll
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_bgn][i]));
    // Lr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_end - 1][i]));
    // Rl
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[rchild_bgn][i]));
    // Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[rchild_end - 1][i]));
    // Rr+
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++],
        (rchild_end < sentence_size) ?
            sentence_fp[rchild_end][i] : kFpEmpty));

    // Ll-&Ll
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++],
        ((lchild_bgn - 1 >= 0) ? sentence_fp[lchild_bgn - 1][i] : kFpEmpty),
        sentence_fp[lchild_bgn][i]));
    // Ll&Lr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_bgn][i],
        sentence_fp[lchild_end - 1][i]));
    // Ll&Rl
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_bgn][i],
        sentence_fp[rchild_bgn][i]));
    // Ll&Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_bgn][i],
        sentence_fp[rchild_end - 1][i]));
    // Lr&Rl
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_end - 1][i],
        sentence_fp[rchild_bgn][i]));
    // Lr&Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_end - 1][i],
        sentence_fp[rchild_end - 1][i]));
    // Rl&Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[rchild_bgn][i],
        sentence_fp[rchild_end - 1][i]));
    // Rr&Rr+
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[rchild_end - 1][i],
        ((rchild_end < sentence_size) ?
            sentence_fp[rchild_end][i] : kFpEmpty)));

    // Lr-&Lr&Rl
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++],
        ((lchild_end - 2 >= 0) ? sentence_fp[lchild_end - 2][i] : kFpEmpty),
        sentence_fp[lchild_end - 1][i],
        sentence_fp[rchild_bgn][i]));
    // Ll&Lr&Rl
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_bgn][i],
        sentence_fp[lchild_end - 1][i], sentence_fp[rchild_bgn][i]));
    // Lr&Rl&Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_end - 1][i],
        sentence_fp[rchild_bgn][i], sentence_fp[rchild_end - 1][i]));

    // Lr&Rl&Rl+
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_end - 1][i],
        sentence_fp[rchild_bgn][i],
        ((rchild_bgn + 1 < sentence_size) ?
            sentence_fp[rchild_bgn + 1][i] : kFpEmpty)));

    // Ll&Lr&Rl&Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], sentence_fp[lchild_bgn][i],
        sentence_fp[lchild_end - 1][i], sentence_fp[rchild_bgn][i],
        sentence_fp[rchild_end - 1][i]));

    // NT&SIDE&Ll
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], kFpNumbers[gp_nt], kFpNumbers[gp_side],
        sentence_fp[lchild_bgn][i]));
    // NT&SIDE&Lr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], kFpNumbers[gp_nt], kFpNumbers[gp_side],
        sentence_fp[lchild_end - 1][i]));
    // NT&SIDE&Rl
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], kFpNumbers[gp_nt], kFpNumbers[gp_side],
        sentence_fp[rchild_bgn][i]));
    // NT&SIDE&Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], kFpNumbers[gp_nt], kFpNumbers[gp_side],
        sentence_fp[rchild_end - 1][i]));

    // NT&SIDE&Ll&Rr
    features->push_back(FeatureFingerprintCat(
        kFpNumbers[id++], kFpNumbers[gp_nt], kFpNumbers[gp_side],
        sentence_fp[lchild_bgn][i], sentence_fp[rchild_end - 1][i]));
  }

  CHECK(features->size() <= kMaxFeatures, "Too many features");
}
