/*!
 *  Copyright (c) 2019 by Contributors
 * \file random/choice.cc
 * \brief Non-uniform discrete sampling implementation
 */

#include <dgl/random.h>
#include <dgl/array.h>
#include <vector>
#include <numeric>
#include "sample_utils.h"
#include "../third_party/phmap/parallel_hashmap/phmap.h"


template<typename T>
struct Value {
phmap::flat_hash_set<T> selected;
T *out;
}; 
template<typename T>
static std::map<std::string, std::shared_ptr<Value<T>>> cache_;

namespace dgl {

template<typename IdxType>
IdxType RandomEngine::Choice(FloatArray prob) {
  IdxType ret = 0;
  ATEN_FLOAT_TYPE_SWITCH(prob->dtype, ValueType, "probability", {
    // TODO(minjie): allow choosing different sampling algorithms
    utils::TreeSampler<IdxType, ValueType, true> sampler(this, prob);
    ret = sampler.Draw();
  });
  return ret;
}

template int32_t RandomEngine::Choice<int32_t>(FloatArray);
template int64_t RandomEngine::Choice<int64_t>(FloatArray);


template<typename IdxType, typename FloatType>
void RandomEngine::Choice(IdxType num, FloatArray prob, IdxType* out, bool replace) {
  const IdxType N = prob->shape[0];
  if (!replace)
    CHECK_LE(num, N) << "Cannot take more sample than population when 'replace=false'";
  if (num == N && !replace)
    std::iota(out, out + num, 0);

  utils::BaseSampler<IdxType>* sampler = nullptr;
  if (replace) {
    sampler = new utils::TreeSampler<IdxType, FloatType, true>(this, prob);
  } else {
    sampler = new utils::TreeSampler<IdxType, FloatType, false>(this, prob);
  }
  for (IdxType i = 0; i < num; ++i)
    out[i] = sampler->Draw();
  delete sampler;
}

template void RandomEngine::Choice<int32_t, float>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, float>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, double>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, double>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);

template <typename IdxType>
void RandomEngine::UniformChoice(IdxType num, IdxType population, IdxType* out, bool replace) {
  if (!replace)
    CHECK_LE(num, population) << "Cannot take more sample than population when 'replace=false'";
  if (replace) {
    for (IdxType i = 0; i < num; ++i)
      out[i] = RandInt(population);
  } else {
    if (num < population / 10) {  // TODO(minjie): may need a better threshold here
      // use hash set
      // In the best scenario, time complexity is O(num), i.e., no conflict.
      //
      // Let k be num / population, the expected number of extra sampling steps is roughly
      // k^2 / (1-k) * population, which means in the worst case scenario,
      // the time complexity is O(population^2). In practice, we use 1/10 since
      //std::unordered_set is pretty slow.
      //std::unordered_set<IdxType> selected;
      std::string hash_key = "" + std::to_string(num);
      hash_key += std::to_string(population);
      hash_key += std::to_string(replace);
      hash_key += std::to_string(*out);
      auto hash_key1 = std::to_string(std::hash<std::string>()(hash_key)); 
      phmap::flat_hash_set<IdxType> selected;
      if(cache_<IdxType>.find(hash_key1) == cache_<IdxType>.end()) {
	while (selected.size() < num) {
        	selected.insert(RandInt(population));
      	}
	std::shared_ptr<Value<IdxType>> shared = std::make_shared<Value<IdxType>>();
	shared->selected = selected;
	cache_<IdxType>[hash_key1] = shared;
      } else {
	std::shared_ptr<Value<IdxType>> cache_local = cache_<IdxType>[hash_key1];
	selected = cache_local->selected;
      }
      std::copy(selected.begin(), selected.end(), out);
    } else {
      // reservoir algorithm
      // time: O(population), space: O(num)
      for (IdxType i = 0; i < num; ++i)
        out[i] = i;
      for (IdxType i = num; i < population; ++i) {
        const IdxType j = RandInt(i + 1);
        if (j < num)
          out[j] = i;
      }
    }
  }
}

template void RandomEngine::UniformChoice<int32_t>(
    int32_t num, int32_t population, int32_t* out, bool replace);
template void RandomEngine::UniformChoice<int64_t>(
    int64_t num, int64_t population, int64_t* out, bool replace);

};  // namespace dgl
