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

// Miscellaneous utility functions.

#ifndef MISC_H_
#define MISC_H_

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define CHECK(condition, message) do { \
  if (!(condition)) { \
    std::cerr << "ERROR(" << __FILE__ << ":" << __LINE__ << ") " \
              << (message) << std::endl; \
    abort(); \
  } \
} while (0)

inline std::vector<std::string> Split(const std::string &str,
                                      const std::string &delim) {
  std::vector<std::string> result;
  std::string::size_type p, q;
  for (p = 0; (q = str.find(delim, p)) != std::string::npos;
       p = q + delim.size()) {
    result.emplace_back(str, p, q - p);
  }
  result.emplace_back(str, p);
  return result;
}

inline std::string Join(const std::vector<std::string> &strs,
                        const std::string &delim) {
  std::string result;
  if (!strs.empty()) {
    result.append(strs[0]);
  }
  for (size_t i = 1; i < strs.size(); i++) {
    result.append(delim);
    result.append(strs[i]);
  }
  return result;
}

inline void Replace(const std::string &src, const std::string &dst,
                    std::string *str) {
  std::string::size_type pos = 0;
  while ((pos = str->find(src, pos)) != std::string::npos) {
    str->replace(pos, src.size(), dst);
    pos += dst.size();
  }
}

#endif  // MISC_H_
