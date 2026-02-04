#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        int rankA = inputs[0]->getRank();
        int rankB = inputs[1]->getRank();
    
        int m = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
        int ka = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];
        int kb = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        int n = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];
    
        IT_ASSERT(ka == kb);
    
        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);
        Shape out = infer_broadcast(batchA, batchB);
    
        out.push_back(m);
        out.push_back(n);
    
        return {{out}};
        // =================================== 作业 ===================================
        return std::nullopt;
    }

} // namespace infini