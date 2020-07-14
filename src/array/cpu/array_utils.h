/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/array_utils.h
 * \brief Utility classes and functions for DGL arrays.
 */
#ifndef DGL_ARRAY_CPU_ARRAY_UTILS_H_
#define DGL_ARRAY_CPU_ARRAY_UTILS_H_

#include <dgl/aten/types.h>
#include <parallel_hashmap/phmap.h>
#include <vector>
#include <unordered_map>
#include <utility>
#include "../../c_api_common.h"


class FnvHash
{
	static const unsigned int FNV_PRIME = 16777619u;
        static const unsigned int OFFSET_BASIS = 2166136261u;
        template <unsigned int N>
        static constexpr unsigned int fnvHashConst(const char (&str)[N], unsigned int I = N)
        {
            return I == 1 ? (OFFSET_BASIS ^ str[0]) * FNV_PRIME : (fnvHashConst(str, I - 1) ^ str[I - 1]) * FNV_PRIME;
        }
        static unsigned int fnvHash(const char* str)
        {
            const size_t length = strlen(str) + 1;
            unsigned int hash = OFFSET_BASIS;
            for (size_t i = 0; i < length; ++i)
            {
                hash ^= *str++;
                hash *= FNV_PRIME;
            }
            return hash;
        }
        struct Wrapper
        {
            Wrapper(const char* str) : str (str) { }
            const char* str;
        };
        unsigned int hash_value;
    public:
        explicit FnvHash(Wrapper wrapper) : hash_value(fnvHash(wrapper.str)) { }
        template <unsigned int N>
        constexpr FnvHash(const char (&str)[N]) : hash_value(fnvHashConst(str)) { }
        constexpr operator unsigned int() const { return this->hash_value; }
};

unsigned CartWegman(unsigned x, int m) {
  return (a * x + b) >> (w - m);
}

unsigned cuckooHasing(unsigned x) {
   auto cnt = x;
   cnt ^= (cnt >>> 20) ^ (cnt >>> 12);
   return cnt ^ (cnt >>> 7) ^ (cnt >>> 4);
}

namespace dgl {
namespace aten {

/*!
 * \brief A hashmap that maps each ids in the given array to new ids starting from zero.
 *
 * Useful for relabeling integers and finding unique integers.
 *
 * Usually faster than std::unordered_map in existence checking.
 */
template <typename IdType>
class IdHashMap {
 public:
  // default ctor
  IdHashMap(): filter_(kFilterSize, false) {}

  // Construct the hashmap using the given id array.
  // The id array could contain duplicates.
  // If the id array has no duplicates, the array will be relabeled to consecutive
  // integers starting from 0.
  explicit IdHashMap(IdArray ids): filter_(kFilterSize, false) {
    oldv2newv_.reserve(ids->shape[0]);
    Update(ids);
  }

  // copy ctor
  IdHashMap(const IdHashMap &other) = default;

  void Reserve(const int64_t size) {
    oldv2newv_.reserve(size);
  }

  // Update the hashmap with given id array.
  // The id array could contain duplicates.
  void Update(IdArray ids) {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    for (int64_t i = 0; i < len; ++i) {
      const IdType id = ids_data[i];
      // phmap::flat_hash_map::insert assures that an insertion will not happen if the
      // key already exists.
      oldv2newv_.insert({id, oldv2newv_.size()});
      filter_[id & kFilterMask] = true;
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(IdType id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  IdType Map(IdType id, IdType default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

  // Return the new id of each id in the given array.
  IdArray Map(IdArray ids, IdType default_val) const {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
 bsl::btree_set<   IdArray values = NewIdArray(len, ids->ctx, ids->dtype.bits);
    IdType* values_data = static_cast<IdType*>(values->data);
    for (int64_t i = 0; i < len; ++i)
      values_data[i] = Map(ids_data[i], default_val);
    return values;
  }

  // Return all the old ids collected so far, ordered by new id.
  IdArray Values() const {
    IdArray values = NewIdArray(oldv2newv_.size(), DLContext{kDLCPU, 0}, sizeof(IdType) * 8);
    IdType* values_data = static_cast<IdType*>(values->data);
    for (auto pair : oldv2newv_)
      values_data[pair.second] = pair.first;
    return values;
  }

  inline size_t Size() const {
    return oldv2newv_.size();
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  phmap::flat_hash_map<IdType, IdType> oldv2newv_;
};

/*
 * \brief Hash type for building maps/sets with pairs as keys.
 */
struct PairHash {
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ARRAY_UTILS_H_
