#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <iostream>

namespace infini
{
    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";
        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted) return true;
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified) return false;
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = ops.begin(); it != ops.end(); ) {
                auto op = *it;
                bool removed = false;
    
                if (op->getOpType() == OpType::MatMul) {
                    auto matmul = std::dynamic_pointer_cast<MatmulObj>(op);
                    auto inputs = matmul->getInputs();
                    for (int i = 0; i < (int)inputs.size(); ++i) {
                        auto inT = inputs[i];
                        auto preOp = inT->getSource();
                        if (preOp && preOp->getOpType() == OpType::Transpose) {
                            auto trans = std::dynamic_pointer_cast<TransposeObj>(preOp);
                            auto perm = trans->getPermute();
                            int r = perm.size();
                            if (r >= 2 && perm[r-1] == r-2 && perm[r-2] == r-1) {
                                bool isSwap = true;
                                for (int j = 0; j < r - 2; ++j) if (perm[j] != j) isSwap = false;
                                if (isSwap) {
                                    auto startInT = trans->getInputs()[0];
                                    if (i == 0) matmul->setTransA(!matmul->getTransA());
                                    else matmul->setTransB(!matmul->getTransB());
    
                                    matmul->replaceInput(inT, startInT);
                                    inT->removeTarget(op);
                                    startInT->addTarget(op);
                                    op->removePredecessors(preOp);
                                    if (auto startOp = startInT->getSource()) {
                                        op->addPredecessors(startOp);
                                        startOp->addSuccessors(op);
                                    }
                                    changed = true;
                                }
                            }
                        }
                    }
                }
    
                // 规则 2: 
                if (!changed && op->getOpType() == OpType::Transpose) {
                    auto inT = op->getInputs()[0];
                    auto preOp = inT->getSource();
                    if (preOp && preOp->getOpType() == OpType::Transpose) {
                        auto trans1 = std::dynamic_pointer_cast<TransposeObj>(preOp);
                        auto trans2 = std::dynamic_pointer_cast<TransposeObj>(op);
                        auto p1 = trans1->getPermute();
                        auto p2 = trans2->getPermute();
                        bool isId = (p1.size() == p2.size());
                        if (isId) {
                            for (int j = 0; j < (int)p1.size(); ++j) if (p1[p2[j]] != j) isId = false;
                            if (isId) {
                                auto outT = op->getOutput();
                                auto startInT = trans1->getInputs()[0];
                                auto targets = outT->getTargets();
                                std::vector<Operator> targetVec(targets.begin(), targets.end());
                                for (auto &t : targetVec) {
                                    t->replaceInput(outT, startInT);
                                    outT->removeTarget(t);
                                    startInT->addTarget(t);
                                    t->removePredecessors(op);
                                    if (auto startOp = startInT->getSource()) {
                                        t->addPredecessors(startOp);
                                        startOp->addSuccessors(t);
                                    }
                                }
                                changed = true;
                            }
                        }
                    }
                }
    
                auto out = op->getOutput();
                if (out && out->getTargets().empty() && out != tensors.back()) {
                    for (auto &in : op->getInputs()) { if (in) in->removeTarget(op); }
                    it = ops.erase(it);
                    changed = true;
                    removed = true;
                }
                if (!removed) ++it;
            }

            for (auto it = tensors.begin(); it != tensors.end(); ) {
                if ((*it)->getTargets().empty() && !(*it)->getSource() && *it != tensors.back()) {
                    it = tensors.erase(it);
                    changed = true;
                } else ++it;
            }
        }
    }
  
   
   
   
   
    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors) if (tensor->getFuid() == fuid) return tensor;
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                if (newShape != oldOutputs[i]->getDims())
                {
                    auto tensor = this->getTensor(oldOutputs[i]->getFuid());
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        IT_ASSERT(topo_sort() == true);
        std::map<Tensor, size_t> offsets;
        for (auto &tensor : tensors) offsets[tensor] = allocator.alloc(tensor->getBytes());
        void *basePtr = allocator.getPtr();
        for (auto &tensor : tensors) {
            size_t offset = offsets[tensor];
            uint8_t *realPtr = static_cast<uint8_t *>(basePtr) + offset;
            tensor->setDataBlob(make_ref<BlobObj>(runtime, realPtr));
        }
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime);
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors) addTensor(t);
        return tensors;
    }

    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets()) IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs()) IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
            for (auto tensor : op->getOutputs()) IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
        }
        return true;
    }
}