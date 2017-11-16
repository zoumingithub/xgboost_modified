/*!
 * Copyright 2015 by Contributors
 * \file rank.cc
 * \brief Definition of rank loss.
 * \author Tianqi Chen, Kailong Chen
 */
#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../common/math.h"
#include "../common/random.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(rank_obj);

struct LambdaRankParam : public dmlc::Parameter<LambdaRankParam> {
  int num_pairsample;
  float fix_list_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LambdaRankParam) {
    DMLC_DECLARE_FIELD(num_pairsample).set_lower_bound(1).set_default(1)
        .describe("Number of pair generated for each instance.");
    DMLC_DECLARE_FIELD(fix_list_weight).set_lower_bound(0.0f).set_default(0.0f)
        .describe("Normalize the weight of each list by this value,"
                  " if equals 0, no effect will happen");
  }
};

// objective for lambda rank
class LambdaRankObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }

  inline bst_float EvalLogLoss(bst_float y, bst_float py) const {
    const bst_float eps = 1e-16f;
    const bst_float pneg = 1.0f - py;
    if (py < eps) {
      return -y * std::log(eps) - (1.0f - y)  * std::log(1.0f - eps);
    } else if (pneg < eps) {
      return -y * std::log(1.0f - eps) - (1.0f - y)  * std::log(eps);
    } else {
      return -y * std::log(py) - (1.0f - y) * std::log(pneg);
    }
  }
 
  void GetGradient(const std::vector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   std::vector<bst_gpair>* out_gpair) override {
    CHECK_EQ(preds.size(), info.rankpairs.size()) << "pairs number predict size not match";
    std::vector<bst_gpair>& gpair = *out_gpair;
    gpair.resize(preds.size());

    const bst_omp_uint docnum = static_cast<bst_omp_uint>(info.rankpairs.size());
    for (bst_omp_uint docid = 0; docid < docnum; ++docid) {
        gpair[docid] = bst_gpair(0.0f,0.0f);
    }

    double total_pair_loss = 0;
    double total_cls_loss = 0;
    double total_weight = 0;

    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(info.labels.size());
    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    CHECK(gptr.size() != 0 && gptr.back() == info.labels.size())
        << "group structure not consistent with #rows";
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);

    #pragma omp parallel
    {
      // parall construct, declare random number generator here, so that each
      // thread use its own random number generator, seed by thread id and current iteration
      common::RandomEngine rnd(iter * 1111 + omp_get_thread_num());
      std::vector<LambdaPair> pairs;
      std::vector< std::pair<bst_float, unsigned> > rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
          for (bst_omp_uint docid = gptr[k]; docid < gptr[k+1]; ++docid) {
            for (size_t i = 0; i < info.rankpairs[docid].size(); ++i) {
              size_t cmp_docid = info.rankpairs[docid][i];
              //float weight = common::Sigmoid(info.pairweight[docid][i]/30.0);
              float weight = info.pairweight[docid][i];
              int pos_id, neg_id;
              if(info.pairweight[docid][i]>0) 
              {
                 pos_id = docid;
                 neg_id = cmp_docid;
              }
              else{
                pos_id = cmp_docid;
                neg_id = docid;
              }

              //if(docid<=100) std::cout << "pos id "<<pos_id+1 << " neg id"<<neg_id+1<<std::endl;
              const float alpha = 0.5f;
              const float sigma = 100;
              bst_float pos_pred = preds[pos_id];
              bst_float neg_pred = preds[neg_id];
              const float eps = 1e-16f;
              const float w = 1.0f;//std::max((float)eps,(float)std::fabs(weight));         
              bst_float p = common::Sigmoid(sigma*(pos_pred - neg_pred));
              bst_float g = sigma*(p - 1.0f);
              bst_float h = sigma*std::max(p * (1.0f - p), eps);
              // accumulate gradient and hessian in both pid, and nid
              
              gpair[pos_id].grad += alpha*g * w;
              gpair[pos_id].hess += alpha*2.0f * w * h;
              gpair[neg_id].grad -= alpha*g * w;
              gpair[neg_id].hess += alpha*2.0f * w * h;
              float pos_label = info.labels[pos_id]>0?1:0;
              float neg_label = info.labels[neg_id]>0?1:0;
              gpair[pos_id].grad += 0.5*(1-alpha)*w*(pos_pred - pos_label);
              gpair[neg_id].grad += 0.5*(1-alpha)*w*(neg_pred - neg_label);
              gpair[pos_id].hess += 0.5*(1-alpha)*w*std::max(pos_pred*(1-pos_pred),eps);
              gpair[neg_id].hess += 0.5*(1-alpha)*w*std::max(neg_pred*(1-neg_pred),eps);
              //std::cout << "pair-grad  " << alpha*g * w << " classification grad "<<0.5*(1-alpha)*w*(pos_pred - pos_label);
              //compute pairloss here
              float pair_loss = EvalLogLoss(1.0f,sigma*(pos_pred - neg_pred));
              float cls_loss = 0;
              cls_loss += 0.5*EvalLogLoss(pos_label,pos_pred);
              cls_loss += 0.5*EvalLogLoss(neg_label,neg_pred);
              total_pair_loss += pair_loss;
              total_cls_loss += cls_loss;
              total_weight += w;
            }
        }
      }
    }
    std::cout << "pair loss  " << total_pair_loss/total_weight << " cls loss "<<total_cls_loss/total_weight<<std::endl;
  }
  const char* DefaultEvalMetric(void) const override {
    return "map";
  }

 protected:
  /*! \brief helper information in a list */
  struct ListEntry {
    /*! \brief the predict score we in the data */
    bst_float pred;
    /*! \brief the actual label of the entry */
    bst_float label;
    /*! \brief row index in the data matrix */
    unsigned rindex;
    // constructor
    ListEntry(bst_float pred, bst_float label, unsigned rindex)
        : pred(pred), label(label), rindex(rindex) {}
    // comparator by prediction
    inline static bool CmpPred(const ListEntry &a, const ListEntry &b) {
      return a.pred > b.pred;
    }
    // comparator by label
    inline static bool CmpLabel(const ListEntry &a, const ListEntry &b) {
      return a.label > b.label;
    }
  };
  /*! \brief a pair in the lambda rank */
  struct LambdaPair {
    /*! \brief positive index: this is a position in the list */
    unsigned pos_index;
    /*! \brief negative index: this is a position in the list */
    unsigned neg_index;
    /*! \brief weight to be filled in */
    bst_float weight;
    // constructor
    LambdaPair(unsigned pos_index, unsigned neg_index)
        : pos_index(pos_index), neg_index(neg_index), weight(1.0f) {}
  };
  /*!
   * \brief get lambda weight for existing pairs
   * \param list a list that is sorted by pred score
   * \param io_pairs record of pairs, containing the pairs to fill in weights
   */
  virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                               std::vector<LambdaPair> *io_pairs) = 0;

 private:
  LambdaRankParam param_;
};

// objective for lambda rank
class LambdaRankObj_BKP : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   std::vector<bst_gpair>* out_gpair) override {
    CHECK_EQ(preds.size(), info.labels.size()) << "label size predict size not match";
    std::vector<bst_gpair>& gpair = *out_gpair;
    gpair.resize(preds.size());
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(info.labels.size());
    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    CHECK(gptr.size() != 0 && gptr.back() == info.labels.size())
        << "group structure not consistent with #rows";
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    #pragma omp parallel
    {
      // parall construct, declare random number generator here, so that each
      // thread use its own random number generator, seed by thread id and current iteration
      common::RandomEngine rnd(iter * 1111 + omp_get_thread_num());

      std::vector<LambdaPair> pairs;
      std::vector<ListEntry>  lst;
      std::vector< std::pair<bst_float, unsigned> > rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        lst.clear(); pairs.clear();
        for (unsigned j = gptr[k]; j < gptr[k+1]; ++j) {
          lst.push_back(ListEntry(preds[j], info.labels[j], j));
          gpair[j] = bst_gpair(0.0f, 0.0f);
        }
        std::sort(lst.begin(), lst.end(), ListEntry::CmpPred);
        rec.resize(lst.size());
        for (unsigned i = 0; i < lst.size(); ++i) {
          rec[i] = std::make_pair(lst[i].label, i);
        }
        std::sort(rec.begin(), rec.end(), common::CmpFirst);
        // enumerate buckets with same label, for each item in the lst, grab another sample randomly
        for (unsigned i = 0; i < rec.size(); ) {
          unsigned j = i + 1;
          while (j < rec.size() && rec[j].first == rec[i].first) ++j;
          // bucket in [i,j), get a sample outside bucket
          unsigned nleft = i, nright = static_cast<unsigned>(rec.size() - j);
          if (nleft + nright != 0) {
            int nsample = param_.num_pairsample;
            while (nsample --) {
              for (unsigned pid = i; pid < j; ++pid) {
                unsigned ridx = std::uniform_int_distribution<unsigned>(0, nleft + nright - 1)(rnd);
                if (ridx < nleft) {
                  pairs.push_back(LambdaPair(rec[ridx].second, rec[pid].second));
                } else {
                  pairs.push_back(LambdaPair(rec[pid].second, rec[ridx+j-i].second));
                }
              }
            }
          }
          i = j;
        }
        // get lambda weight for the pairs
        this->GetLambdaWeight(lst, &pairs);
        // rescale each gradient and hessian so that the lst have constant weighted
        float scale = 1.0f / param_.num_pairsample;
        if (param_.fix_list_weight != 0.0f) {
          scale *= param_.fix_list_weight / (gptr[k + 1] - gptr[k]);
        }
        for (size_t i = 0; i < pairs.size(); ++i) {
          const ListEntry &pos = lst[pairs[i].pos_index];
          const ListEntry &neg = lst[pairs[i].neg_index];
          const bst_float w = pairs[i].weight * scale;
          const float eps = 1e-16f;
          bst_float p = common::Sigmoid(pos.pred - neg.pred);
          bst_float g = p - 1.0f;
          bst_float h = std::max(p * (1.0f - p), eps);
          // accumulate gradient and hessian in both pid, and nid
          gpair[pos.rindex].grad += g * w;
          gpair[pos.rindex].hess += 2.0f * w * h;
          gpair[neg.rindex].grad -= g * w;
          gpair[neg.rindex].hess += 2.0f * w * h;
        }
      }
    }
  }
  const char* DefaultEvalMetric(void) const override {
    return "map";
  }

 protected:
  /*! \brief helper information in a list */
  struct ListEntry {
    /*! \brief the predict score we in the data */
    bst_float pred;
    /*! \brief the actual label of the entry */
    bst_float label;
    /*! \brief row index in the data matrix */
    unsigned rindex;
    // constructor
    ListEntry(bst_float pred, bst_float label, unsigned rindex)
        : pred(pred), label(label), rindex(rindex) {}
    // comparator by prediction
    inline static bool CmpPred(const ListEntry &a, const ListEntry &b) {
      return a.pred > b.pred;
    }
    // comparator by label
    inline static bool CmpLabel(const ListEntry &a, const ListEntry &b) {
      return a.label > b.label;
    }
  };
  /*! \brief a pair in the lambda rank */
  struct LambdaPair {
    /*! \brief positive index: this is a position in the list */
    unsigned pos_index;
    /*! \brief negative index: this is a position in the list */
    unsigned neg_index;
    /*! \brief weight to be filled in */
    bst_float weight;
    // constructor
    LambdaPair(unsigned pos_index, unsigned neg_index)
        : pos_index(pos_index), neg_index(neg_index), weight(1.0f) {}
  };
  /*!
   * \brief get lambda weight for existing pairs
   * \param list a list that is sorted by pred score
   * \param io_pairs record of pairs, containing the pairs to fill in weights
   */
  virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                               std::vector<LambdaPair> *io_pairs) = 0;

 private:
  LambdaRankParam param_;
};

class PairwiseRankObj: public LambdaRankObj{
 protected:
  void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                       std::vector<LambdaPair> *io_pairs) override {}
};

// beta version: NDCG lambda rank
class LambdaRankObjNDCG : public LambdaRankObj {
 protected:
  void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                       std::vector<LambdaPair> *io_pairs) override {
    std::vector<LambdaPair> &pairs = *io_pairs;
    float IDCG;
    {
      std::vector<bst_float> labels(sorted_list.size());
      for (size_t i = 0; i < sorted_list.size(); ++i) {
        labels[i] = sorted_list[i].label;
      }
      std::sort(labels.begin(), labels.end(), std::greater<bst_float>());
      IDCG = CalcDCG(labels);
    }
    if (IDCG == 0.0) {
      for (size_t i = 0; i < pairs.size(); ++i) {
        pairs[i].weight = 0.0f;
      }
    } else {
      IDCG = 1.0f / IDCG;
      for (size_t i = 0; i < pairs.size(); ++i) {
        unsigned pos_idx = pairs[i].pos_index;
        unsigned neg_idx = pairs[i].neg_index;
        float pos_loginv = 1.0f / std::log2(pos_idx + 2.0f);
        float neg_loginv = 1.0f / std::log2(neg_idx + 2.0f);
        int pos_label = static_cast<int>(sorted_list[pos_idx].label);
        int neg_label = static_cast<int>(sorted_list[neg_idx].label);
        bst_float original =
            ((1 << pos_label) - 1) * pos_loginv + ((1 << neg_label) - 1) * neg_loginv;
        float changed  =
            ((1 << neg_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * neg_loginv;
        bst_float delta = (original - changed) * IDCG;
        if (delta < 0.0f) delta = - delta;
        pairs[i].weight = delta;
      }
    }
  }
  inline static bst_float CalcDCG(const std::vector<bst_float> &labels) {
    double sumdcg = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
      const unsigned rel = static_cast<unsigned>(labels[i]);
      if (rel != 0) {
        sumdcg += ((1 << rel) - 1) / std::log2(static_cast<bst_float>(i + 2));
      }
    }
    return static_cast<bst_float>(sumdcg);
  }
};

class LambdaRankObjMAP : public LambdaRankObj {
 protected:
  struct MAPStats {
    /*! \brief the accumulated precision */
    float ap_acc;
    /*!
     * \brief the accumulated precision,
     *   assuming a positive instance is missing
     */
    float ap_acc_miss;
    /*!
     * \brief the accumulated precision,
     * assuming that one more positive instance is inserted ahead
     */
    float ap_acc_add;
    /* \brief the accumulated positive instance count */
    float hits;
    MAPStats(void) {}
    MAPStats(float ap_acc, float ap_acc_miss, float ap_acc_add, float hits)
        : ap_acc(ap_acc), ap_acc_miss(ap_acc_miss), ap_acc_add(ap_acc_add), hits(hits) {}
  };
  /*!
   * \brief Obtain the delta MAP if trying to switch the positions of instances in index1 or index2
   *        in sorted triples
   * \param sorted_list the list containing entry information
   * \param index1,index2 the instances switched
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline bst_float GetLambdaMAP(const std::vector<ListEntry> &sorted_list,
                                int index1, int index2,
                                std::vector<MAPStats> *p_map_stats) {
    std::vector<MAPStats> &map_stats = *p_map_stats;
    if (index1 == index2 || map_stats[map_stats.size() - 1].hits == 0) {
      return 0.0f;
    }
    if (index1 > index2) std::swap(index1, index2);
    bst_float original = map_stats[index2].ap_acc;
    if (index1 != 0) original -= map_stats[index1 - 1].ap_acc;
    bst_float changed = 0;
    bst_float label1 = sorted_list[index1].label > 0.0f ? 1.0f : 0.0f;
    bst_float label2 = sorted_list[index2].label > 0.0f ? 1.0f : 0.0f;
    if (label1 == label2) {
      return 0.0;
    } else if (label1 < label2) {
      changed += map_stats[index2 - 1].ap_acc_add - map_stats[index1].ap_acc_add;
      changed += (map_stats[index1].hits + 1.0f) / (index1 + 1);
    } else {
      changed += map_stats[index2 - 1].ap_acc_miss - map_stats[index1].ap_acc_miss;
      changed += map_stats[index2].hits / (index2 + 1);
    }
    bst_float ans = (changed - original) / (map_stats[map_stats.size() - 1].hits);
    if (ans < 0) ans = -ans;
    return ans;
  }
  /*
   * \brief obtain preprocessing results for calculating delta MAP
   * \param sorted_list the list containing entry information
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline void GetMAPStats(const std::vector<ListEntry> &sorted_list,
                          std::vector<MAPStats> *p_map_acc) {
    std::vector<MAPStats> &map_acc = *p_map_acc;
    map_acc.resize(sorted_list.size());
    bst_float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (size_t i = 1; i <= sorted_list.size(); ++i) {
      if (sorted_list[i - 1].label > 0.0f) {
        hit++;
        acc1 += hit / i;
        acc2 += (hit - 1) / i;
        acc3 += (hit + 1) / i;
      }
      map_acc[i - 1] = MAPStats(acc1, acc2, acc3, hit);
    }
  }
  void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                       std::vector<LambdaPair> *io_pairs) override {
    std::vector<LambdaPair> &pairs = *io_pairs;
    std::vector<MAPStats> map_stats;
    GetMAPStats(sorted_list, &map_stats);
    for (size_t i = 0; i < pairs.size(); ++i) {
      pairs[i].weight =
          GetLambdaMAP(sorted_list, pairs[i].pos_index,
                       pairs[i].neg_index, &map_stats);
    }
  }
};

// register the objective functions
DMLC_REGISTER_PARAMETER(LambdaRankParam);

XGBOOST_REGISTER_OBJECTIVE(PairwiseRankObj, "rank:pairwise")
.describe("Pairwise rank objective.")
.set_body([]() { return new PairwiseRankObj(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankNDCG, "rank:ndcg")
.describe("LambdaRank with NDCG as objective.")
.set_body([]() { return new LambdaRankObjNDCG(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankObjMAP, "rank:map")
.describe("LambdaRank with MAP as objective.")
.set_body([]() { return new LambdaRankObjMAP(); });

}  // namespace obj
}  // namespace xgboost
