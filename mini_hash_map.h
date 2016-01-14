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

// Memory efficient hash map class.
// Key and Val are the types of keys and values stored in the hash map.
// The API of this class is basically a subset of unordered_map.
// However, this class does not support the erase() operation, and a sentinel
// (a value which is never inserted) needs to be specified when constructed.

#ifndef MINI_HASH_MAP_H_
#define MINI_HASH_MAP_H_

#include <cstring>
#include <iterator>
#include <string>
#include <vector>

template<class Key, class Val, class Hash = std::hash<Key> >
class MiniHashMap {
 public:
  typedef size_t size_type;

  class const_iterator {
   public:
    typedef std::forward_iterator_tag iterator_category;
    typedef std::pair<Key, Val> value_type;

    const_iterator() {
    }

    const_iterator(const MiniHashMap *hash_map, int addr)
        : hash_map_(hash_map), addr_(addr) {
      if (addr_ < hash_map_->keys_.size()) {
        entry_.first = hash_map_->keys_[addr_];
        entry_.second = hash_map_->vals_[addr_];
      }
    }

    const_iterator(const const_iterator &it)
        : hash_map_(it.hash_map_), addr_(it.addr_), entry_(it.entry_) {
    }

    const_iterator &operator++() {
      addr_++;
      // Skip empty elements.
      while (addr_ < hash_map_->keys_.size() &&
             hash_map_->vals_[addr_] == hash_map_->sentinel_) {
        addr_++;
      }
      if (addr_ < hash_map_->keys_.size()) {
        entry_.first = hash_map_->keys_[addr_];
        entry_.second = hash_map_->vals_[addr_];
      }
      return *this;
    }

    const_iterator &operator++(int /*postfix*/) {
      const_iterator tmp(*this);
      operator++();
      return tmp;
    }

    const value_type *operator->() const {
      return &entry_;
    }

    const value_type &operator*() const {
      return entry_;
    }

    bool operator==(const const_iterator &it) const {
      return (it.hash_map_ == hash_map_ && it.addr_ == addr_);
    }

    bool operator!=(const const_iterator &it) const {
      return (it.hash_map_ != hash_map_ || it.addr_ != addr_);
    }

   private:
    const MiniHashMap *hash_map_;
    size_type addr_;
    value_type entry_;
  };

  explicit MiniHashMap(const Val &sentinel,
                       double max_load_factor = kMaxLoadFactor)
      : sentinel_(sentinel), max_load_factor_(max_load_factor), size_(0),
        keys_(kInitialSize), vals_(kInitialSize, sentinel) {
  }

  ~MiniHashMap() {
  }

  Val &operator[](const Key &key) {
    size_type addr = index(key);
    if (vals_[addr] == sentinel_) {
      const double load_factor = static_cast<double>(size_) / keys_.size();
      if (load_factor > max_load_factor_) {
        reallocate(2 * keys_.size());
        addr = index(key);
      }

      keys_[addr] = key;
      vals_[addr] = Val();
      size_++;
    }
    return vals_[addr];
  }

  size_type size() const {
    return size_;
  }

  void clear() {
    size_ = 0;
    keys_.assign(kInitialSize, Key());
    vals_.assign(kInitialSize, sentinel_);
  }

  // Fit the size of the hash table for keeping just n entries.
  void resize(size_type n) {
    const size_type capacity = static_cast<size_type>(n / max_load_factor_) + 1;
    reallocate(capacity);
  }

  const_iterator begin() const {
    size_type addr = 0;
    // Find the first non-empty element.
    while (addr < keys_.size() && vals_[addr] == sentinel_) {
      addr++;
    }
    return const_iterator(this, addr);
  }

  const_iterator end() const {
    return const_iterator(this, keys_.size());
  }

  const_iterator find(const Key &key) const {
    const size_type addr = index(key);
    if (vals_[addr] == sentinel_) {
      return end();
    }
    return const_iterator(this, addr);
  }

 private:
  static constexpr int kInitialSize = 256;  // Initial size of the hash table.
  static constexpr double kMaxLoadFactor = 0.8;  // Default maximum load factor.
  const Val sentinel_;  // Empty elements have this values.
  const double max_load_factor_;  // Maximum load factor of this hash map.
  size_type size_;  // Number of entries in the hash table.
  std::vector<Key> keys_;  // Keys.
  std::vector<Val> vals_;  // Values.

  // Look up and return the element with the specified key.
  // Linear probing is used for hash collisions.
  size_type index(const Key &key) const {
    size_type addr = Hash()(key) % keys_.size();
    while (vals_[addr] != sentinel_ && keys_[addr] != key) {
      addr++;
      if (addr >= keys_.size()) {
        addr = 0;
      }
    }
    return addr;
  }

  // Resize the hash table.
  void reallocate(size_type capacity) {
    std::vector<Key> old_keys(capacity);
    old_keys.swap(keys_);
    std::vector<Val> old_vals(capacity, sentinel_);
    old_vals.swap(vals_);
    for (size_type i = 0; i < old_keys.size(); i++) {
      if (old_vals[i] == sentinel_) continue;
      const size_type addr = index(old_keys[i]);
      keys_[addr] = old_keys[i];
      vals_[addr] = old_vals[i];
    }
  }
};

#endif  // MINI_HASH_MAP_H_
