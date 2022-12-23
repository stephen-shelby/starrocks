// Copyright 2021-present StarRocks, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>
#include <queue>
#include <utility>

#include "exec/pipeline/exchange/local_exchange_memory_manager.h"
#include "exec/pipeline/source_operator.h"
#include "exec/vectorized/tablet_info.h"
#include "column/column.h"

namespace starrocks::pipeline {

struct LakePartitionKey {
    LakePartitionKey() = default;
    LakePartitionKey(std::shared_ptr<vectorized::Columns> columns_, uint32_t index_)
            : columns(std::move(columns_)), index(index_) {}
    std::shared_ptr<vectorized::Columns> columns;
    uint32_t index = 0;
};

struct LakePartitionKeyComparator {
    // return true if lhs < rhs
    // 'nullptr' is max value, but 'null' is min value
    bool operator()(const std::shared_ptr<LakePartitionKey>& lhs, const std::shared_ptr<LakePartitionKey>& rhs) const {
        if (lhs->columns == nullptr) {
            return false;
        } else if (rhs->columns == nullptr) {
            return true;
        }
        DCHECK_EQ(lhs->columns->size(), rhs->columns->size());
        for (size_t i = 0; i < lhs->columns->size(); ++i) {
            int cmp = (*lhs->columns)[i]->compare_at(lhs->index, rhs->index, *(*rhs->columns)[i], -1);
            if (cmp != 0) {
                return cmp < 0;
            }
        }
        // equal, return false
        return false;
    }
};

class LocalExchangeSourceOperator final : public SourceOperator {
    class PartitionChunk {
    public:
        PartitionChunk(vectorized::ChunkPtr chunk, std::shared_ptr<std::vector<uint32_t>> indexes, const uint32_t from,
                       const uint32_t size)
                : chunk(std::move(chunk)), indexes(std::move(indexes)), from(from), size(size) {}

        PartitionChunk(const PartitionChunk&) = delete;

        PartitionChunk(PartitionChunk&&) = default;

        vectorized::ChunkPtr chunk;
        std::shared_ptr<std::vector<uint32_t>> indexes;
        const uint32_t from;
        const uint32_t size;
    };

    struct PendingPartitionChunks {
        PendingPartitionChunks(std::shared_ptr<std::queue<PartitionChunk>> partition_chunk_queue_, uint32_t index_)
                : partition_chunk_queue(std::move(partition_chunk_queue_)), partition_row_nums(index_) {}

        std::shared_ptr<std::queue<PartitionChunk>> partition_chunk_queue;
        int64_t partition_row_nums;
    };

public:
    LocalExchangeSourceOperator(OperatorFactory* factory, int32_t id, int32_t plan_node_id, int32_t driver_sequence,
                                const std::shared_ptr<LocalExchangeMemoryManager>& memory_manager)
            : SourceOperator(factory, id, "local_exchange_source", plan_node_id, driver_sequence),
              _memory_manager(memory_manager) {}

    Status add_chunk(vectorized::ChunkPtr chunk);

    Status add_chunk(vectorized::ChunkPtr chunk, std::shared_ptr<std::vector<uint32_t>> indexes, uint32_t from,
                     uint32_t size);

    Status add_chunk(vectorized::ChunkPtr chunk, std::shared_ptr<std::vector<uint32_t>> indexes, uint32_t from,
                     uint32_t size, vectorized::Columns& partition_columns, const std::vector<ExprContext*>& _partition_expr_ctxs);

    bool has_output() const override;

    bool is_finished() const override;

    Status set_finished(RuntimeState* state) override;
    Status set_finishing(RuntimeState* state) override {
        std::lock_guard<std::mutex> l(_chunk_lock);
        _is_finished = true;
        return Status::OK();
    }

    StatusOr<vectorized::ChunkPtr> pull_chunk(RuntimeState* state) override;

private:
    vectorized::ChunkPtr _pull_passthrough_chunk(RuntimeState* state);

    vectorized::ChunkPtr _pull_shuffle_chunk(RuntimeState* state);

    vectorized::ChunkPtr _pull_lake_partition_chunk(RuntimeState* state);

    int64_t _lake_partition_max_rows() const {
        int64_t lake_partition_max_rows = 0;
        if (!_lake_partitions.empty()) {
            for (const auto& i : _lake_partitions) {
                lake_partition_max_rows = std::max(i.second.partition_row_nums, lake_partition_max_rows);
            }
        }
        return lake_partition_max_rows;
    };


    PendingPartitionChunks& _max_row_partition_chunks() {
        using it_type = decltype(_lake_partitions)::value_type;
        auto max_it = std::max_element(_lake_partitions.begin(), _lake_partitions.end(), [](const it_type& lhs, const it_type& rhs) {
            return lhs.second.partition_row_nums < rhs.second.partition_row_nums;
        });

        int64_t lake_partition_max_rows = 0;
        if (!_lake_partitions.empty()) {
            for (const auto& i : _lake_partitions) {
                lake_partition_max_rows = std::max(i.second.partition_row_nums, lake_partition_max_rows);
            }
        }

        return max_it->second;
    };

    bool _is_finished = false;
    std::queue<vectorized::ChunkPtr> _full_chunk_queue;
    std::queue<PartitionChunk> _partition_chunk_queue;
    int64_t _partition_rows_num = 0;

    // TODO(KKS): make it lock free
    mutable std::mutex _chunk_lock;
    const std::shared_ptr<LocalExchangeMemoryManager>& _memory_manager;
    std::map<std::shared_ptr<LakePartitionKey>, PendingPartitionChunks, LakePartitionKeyComparator> _lake_partitions;
};

class LocalExchangeSourceOperatorFactory final : public SourceOperatorFactory {
public:
    LocalExchangeSourceOperatorFactory(int32_t id, int32_t plan_node_id,
                                       std::shared_ptr<LocalExchangeMemoryManager> memory_manager)
            : SourceOperatorFactory(id, "local_exchange_source", plan_node_id),
              _memory_manager(std::move(memory_manager)) {}

    ~LocalExchangeSourceOperatorFactory() override = default;

    OperatorPtr create(int32_t degree_of_parallelism, int32_t driver_sequence) override {
        std::shared_ptr<LocalExchangeSourceOperator> source = std::make_shared<LocalExchangeSourceOperator>(
                this, _id, _plan_node_id, driver_sequence, _memory_manager);
        _sources.emplace_back(source.get());
        return source;
    }

    std::vector<LocalExchangeSourceOperator*>& get_sources() { return _sources; }

private:
    std::shared_ptr<LocalExchangeMemoryManager> _memory_manager;
    std::vector<LocalExchangeSourceOperator*> _sources;
};

} // namespace starrocks::pipeline
