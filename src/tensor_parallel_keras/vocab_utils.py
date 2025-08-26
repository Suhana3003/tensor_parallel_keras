from typing import Sequence


def divide(a: int, b: int) -> int:
    return a // b


class VocabUtility:
    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)