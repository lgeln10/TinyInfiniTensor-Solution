#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

		// =================================== 作业 ===================================
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            size_t blockAddr = it->first;
            size_t blockSize = it->second;
            if (blockSize >= size) {
                freeBlocks.erase(it);
                if (blockSize > size) {
                    freeBlocks[blockAddr + size] = blockSize - size;
                }
                used += size;
                return blockAddr;
            }
        }
        size_t addr = peak;
        peak += size;
        used += size;
        return addr;
        // =================================== 作业 ===================================
        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

	// =================================== 作业 ===================================
        auto it = freeBlocks.insert({addr, size}).first;
        auto next = std::next(it);
        if (next != freeBlocks.end() && it->first + it->second == next->first) {
            it->second += next->second;
            freeBlocks.erase(next);
        }
        if (it != freeBlocks.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == it->first) {
                prev->second += it->second;
                freeBlocks.erase(it);
                it = prev;
            }
        }
        while (!freeBlocks.empty()) {
            auto lastIt = std::prev(freeBlocks.end());
            if (lastIt->first + lastIt->second == peak) {
                peak = lastIt->first;
                freeBlocks.erase(lastIt);
            } else {
                break;
            }
        }
        used -= size;
        // =================================== 作业 ===================================
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
