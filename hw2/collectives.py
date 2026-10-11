import torch
import torch.distributed as dist


def _send_receive(send_tensor: torch.Tensor, receive_tensor: torch.Tensor, rank: int, world_size: int) -> None:
    """Одновременная отправка вправо и получение слева без взаимной блокировки."""
    operations = [
        dist.P2POp(dist.isend, send_tensor.contiguous(), (rank + 1) % world_size),
        dist.P2POp(dist.irecv, receive_tensor, (rank - 1) % world_size),
    ]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if tensor.ndim != 1 or tensor.numel() % world_size:
        raise ValueError("Ожидается одномерный тензор с длиной, кратной world_size")
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    if world_size == 1:
        return chunks[0]

    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        receive_index = (rank - step - 2) % world_size
        incoming = torch.empty_like(chunks[receive_index])
        _send_receive(chunks[send_index], incoming, rank, world_size)
        chunks[receive_index].add_(incoming)
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    if world_size == 1:
        return torch.cat(chunks)

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        _send_receive(chunks[send_index], chunks[receive_index], rank, world_size)
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
