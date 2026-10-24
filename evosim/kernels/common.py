"""Deterministic compaction helpers.

All slot allocation goes through exclusive prefix scans over per-element request flags, so the
mapping request -> slot depends only on element order, never on atomic timing. This is what
makes births, deaths, bond creation and genome allocation deterministic.
"""
import warp as wp


@wp.kernel
def _store_total(flags: wp.array(dtype=wp.int32), tickets: wp.array(dtype=wp.int32), n: int,
                 counters: wp.array(dtype=wp.int64), idx: int):
    if n > 0:
        counters[idx] = wp.int64(tickets[n - 1] + flags[n - 1])
    else:
        counters[idx] = wp.int64(0)


def scan(flags, tickets, counters, total_idx):
    """tickets = exclusive_scan(flags); counters[total_idx] = sum(flags)."""
    n = flags.shape[0]
    wp.utils.array_scan(flags, tickets, inclusive=False)
    wp.launch(_store_total, 1, [flags, tickets, n, counters, total_idx], device=flags.device)


@wp.kernel
def push_free(flags: wp.array(dtype=wp.int32), tickets: wp.array(dtype=wp.int32),
              stack: wp.array(dtype=wp.int32), counters: wp.array(dtype=wp.int64), count_idx: int):
    """stack[count + ticket] = i for flagged i. Count is advanced separately (advance_counter)."""
    i = wp.tid()
    if flags[i] != 0:
        base = int(counters[count_idx])
        stack[base + tickets[i]] = i


@wp.kernel
def advance_counter(counters: wp.array(dtype=wp.int64), idx: int, total_idx: int, sign: int):
    counters[idx] = counters[idx] + wp.int64(sign) * counters[total_idx]


@wp.func
def pop_slot(stack: wp.array(dtype=wp.int32), counters: wp.array(dtype=wp.int64), count_idx: int,
             ticket: int) -> int:
    """Slot for the ticket-th pop (before the count is decremented); -1 if exhausted."""
    cnt = int(counters[count_idx])
    if ticket >= cnt:
        return -1
    return stack[cnt - 1 - ticket]


def push_flagged(flags, tickets, stack, counters, count_idx, total_idx):
    scan(flags, tickets, counters, total_idx)
    wp.launch(push_free, flags.shape[0], [flags, tickets, stack, counters, count_idx], device=flags.device)
    wp.launch(advance_counter, 1, [counters, count_idx, total_idx, 1], device=flags.device)


@wp.func
def fx(x: float) -> wp.int64:
    """float -> fixed point (deterministic integer accumulation)."""
    return wp.int64(x * 4294967296.0)


@wp.func
def unfx(x: wp.int64) -> float:
    return float(wp.float64(x) / wp.float64(4294967296.0))
