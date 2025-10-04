#!/usr/bin/env python3
"""Агрегирует IPv4-адреса из журналов маршрутизатора в компактный список подсетей.

Скрипт извлекает адреса из логов, устраняет дубликаты и запускает оптимизационный
алгоритм с жёстким ограничением на количество подсетей. Оптимизация построена на
релаксации Лагранжа двоичного префиксного дерева: для заданного множителя штрафа
алгоритм решает, схлопнуть ли префикс или оставить дочерние узлы. Двоичный поиск
по λ подбирает наименьший набор подсетей, удовлетворяющий лимиту и минимизирующий
число «лишних» (не наблюдавшихся) адресов.

Параметры по умолчанию соответствуют требованиям кейса: ≤64k подсетей, штраф
0.05 за дополнительный адрес и листья /31 (по два IPv4 на лист), что укладывается
в 30-минутный бюджет на типовом железе.
"""

from __future__ import annotations

import argparse
import ipaddress
import math
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

IPv4Int = int
PrefixKey = int  # компактное кодирование битов префикса

PENALTY_PER_EXTRA_DEFAULT = 0.05
PREFIX_LIMIT_DEFAULT = 64_000
BASE_PREFIX_DEFAULT = 31
BINARY_SEARCH_ITERATIONS_DEFAULT = 36
TOLERANCE = 1e-9

IP_REGEX = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")


def parse_unique_ipv4(path: str) -> List[IPv4Int]:
    """Возвращает отсортированный список уникальных IPv4-адресов из логов."""

    unique: set[IPv4Int] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = IP_REGEX.search(line)
            if not match:
                continue
            octet_str = match.group(1).split(".")
            if len(octet_str) != 4:
                continue
            try:
                octets = [int(x) for x in octet_str]
            except ValueError:
                continue
            if any(o < 0 or o > 255 for o in octets):
                continue
            value = (
                (octets[0] << 24)
                | (octets[1] << 16)
                | (octets[2] << 8)
                | octets[3]
            )
            unique.add(value)
    return sorted(unique)


def build_prefix_counts(unique_ips: Iterable[IPv4Int], base_prefix_len: int) -> List[Dict[PrefixKey, int]]:
    """Формирует счётчики адресов для каждого уровня префиксов вплоть до /0."""

    counts: List[Dict[PrefixKey, int]] = [dict() for _ in range(base_prefix_len + 1)]
    shift = 32 - base_prefix_len
    base_map: Dict[PrefixKey, int] = defaultdict(int)
    for value in unique_ips:
        base_map[value >> shift] += 1
    counts[base_prefix_len] = dict(base_map)

    for length in range(base_prefix_len - 1, -1, -1):
        child_counts = counts[length + 1]
        parent_counts: Dict[PrefixKey, int] = defaultdict(int)
        for child_prefix, cnt in child_counts.items():
            parent_counts[child_prefix >> 1] += cnt
        counts[length] = dict(parent_counts)

    return counts


def run_dp(
    counts: List[Dict[PrefixKey, int]],
    penalty_per_extra: float,
    base_prefix_len: int,
    lambda_value: float,
    keep_decisions: bool = False,
) -> Tuple[float, int, List[Dict[PrefixKey, bool]] | None]:
    """Вычисляет значение релаксированной функции при заданном λ.

    Возвращает суммарную стоимость (штраф + λ * количество подсетей), итоговое
    число подсетей и, при необходимости, карту решений по уровням.
    """

    decisions: List[Dict[PrefixKey, bool]] | None = None
    if keep_decisions:
        decisions = [dict() for _ in range(base_prefix_len + 1)]

    size = 1 << (32 - base_prefix_len)
    dp_prev: Dict[PrefixKey, Tuple[float, int]] = {}
    base_decisions = decisions[base_prefix_len] if decisions is not None else None
    for prefix, actual in counts[base_prefix_len].items():
        extra = size - actual
        cost = extra * penalty_per_extra + lambda_value
        dp_prev[prefix] = (cost, 1)
        if base_decisions is not None:
            base_decisions[prefix] = True

    for length in range(base_prefix_len - 1, -1, -1):
        counts_level = counts[length]
        size = 1 << (32 - length)
        dp_curr: Dict[PrefixKey, Tuple[float, int]] = {}
        level_decisions = decisions[length] if decisions is not None else None

        for prefix, actual in counts_level.items():
            left_prefix = prefix << 1
            right_prefix = left_prefix | 1
            left_cost, left_count = dp_prev.get(left_prefix, (0.0, 0))
            right_cost, right_count = dp_prev.get(right_prefix, (0.0, 0))
            children_cost = left_cost + right_cost
            children_count = left_count + right_count

            extra = size - actual
            parent_cost = extra * penalty_per_extra + lambda_value

            choose_parent = False
            if parent_cost + TOLERANCE < children_cost:
                choose_parent = True
            elif abs(parent_cost - children_cost) <= TOLERANCE:
                if 1 <= children_count:
                    choose_parent = True

            if choose_parent:
                dp_curr[prefix] = (parent_cost, 1)
                if level_decisions is not None:
                    level_decisions[prefix] = True
            else:
                dp_curr[prefix] = (children_cost, children_count)
                if level_decisions is not None:
                    level_decisions[prefix] = False

        dp_prev = dp_curr

    root_cost, root_count = dp_prev[0]
    return root_cost, root_count, decisions


def binary_search_lambda(
    counts: List[Dict[PrefixKey, int]],
    penalty_per_extra: float,
    base_prefix_len: int,
    limit: int,
    iterations: int,
) -> Tuple[float, int, float]:
    """Подбирает λ, обеспечивающий оптимальное решение в пределах лимита подсетей."""

    low = 0.0
    high = 1.0
    _, count_high, _ = run_dp(counts, penalty_per_extra, base_prefix_len, high)
    while count_high > limit:
        high *= 2.0
        _, count_high, _ = run_dp(counts, penalty_per_extra, base_prefix_len, high)

    best_penalty = math.inf
    best_count = limit
    best_lambda = high

    for _ in range(iterations):
        mid = (low + high) / 2.0
        cost_mid, count_mid, _ = run_dp(counts, penalty_per_extra, base_prefix_len, mid)
        if count_mid > limit:
            low = mid
            continue
        high = mid
        penalty_mid = cost_mid - mid * count_mid
        if penalty_mid < best_penalty or (
            abs(penalty_mid - best_penalty) <= TOLERANCE and count_mid < best_count
        ):
            best_penalty = penalty_mid
            best_count = count_mid
            best_lambda = mid

    return best_penalty, best_count, best_lambda


def collect_networks(
    counts: List[Dict[PrefixKey, int]],
    decisions: List[Dict[PrefixKey, bool]],
    base_prefix_len: int,
) -> List[ipaddress.IPv4Network]:
    """Восстанавливает список CIDR-префиксов из карты решений."""

    result: List[ipaddress.IPv4Network] = []

    def dfs(level: int, prefix_value: PrefixKey) -> None:
        if level == base_prefix_len:
            network_int = prefix_value << (32 - level)
            result.append(ipaddress.IPv4Network((network_int, level)))
            return
        if decisions[level].get(prefix_value, False):
            network_int = prefix_value << (32 - level)
            result.append(ipaddress.IPv4Network((network_int, level)))
            return
        left_prefix = prefix_value << 1
        right_prefix = left_prefix | 1
        if left_prefix in counts[level + 1]:
            dfs(level + 1, left_prefix)
        if right_prefix in counts[level + 1]:
            dfs(level + 1, right_prefix)

    dfs(0, 0)
    return result


def compute_metrics(
    networks: Iterable[ipaddress.IPv4Network],
    counts: List[Dict[PrefixKey, int]],
    penalty_per_extra: float,
) -> Tuple[int, int, float]:
    """Возвращает количество подсетей, число «лишних» адресов и итоговый штраф."""

    subnet_count = 0
    extra_total = 0
    for network in networks:
        subnet_count += 1
        prefix_len = network.prefixlen
        prefix_value = int(network.network_address) >> (32 - prefix_len)
        actual = counts[prefix_len][prefix_value]
        extra_total += network.num_addresses - actual
    return subnet_count, extra_total, extra_total * penalty_per_extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the log file with IPv4 records")
    parser.add_argument("output", help="Path to write aggregated CIDR prefixes")
    parser.add_argument(
        "--base-prefix",
        type=int,
        default=BASE_PREFIX_DEFAULT,
        help="Leaf prefix length for DP (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=PREFIX_LIMIT_DEFAULT,
        help="Maximum number of CIDR blocks in the output",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=PENALTY_PER_EXTRA_DEFAULT,
        help="Penalty per extra IPv4 address (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=BINARY_SEARCH_ITERATIONS_DEFAULT,
        help="Binary search iterations for lambda (default: %(default)s)",
    )
    args = parser.parse_args()

    if not (0 <= args.base_prefix <= 32):
        parser.error("--base-prefix must be in [0, 32]")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    print("[1/4] Parsing unique IPv4 addresses...", file=sys.stderr)
    unique_ips = parse_unique_ipv4(args.input)
    print(f"\tfound {len(unique_ips):,} unique IPv4 addresses", file=sys.stderr)

    print("[2/4] Building prefix counts...", file=sys.stderr)
    counts = build_prefix_counts(unique_ips, args.base_prefix)

    print("[3/4] Searching optimal lambda...", file=sys.stderr)
    best_penalty, best_count, best_lambda = binary_search_lambda(
        counts,
        args.penalty,
        args.base_prefix,
        args.limit,
        args.iterations,
    )
    print(
        f"\tbest lambda={best_lambda:.9f}, prefixes={best_count}, "
        f"penalty={best_penalty:,.2f}",
        file=sys.stderr,
    )

    _, _, decisions = run_dp(
        counts,
        args.penalty,
        args.base_prefix,
        best_lambda,
        keep_decisions=True,
    )
    assert decisions is not None

    print("[4/4] Reconstructing subnet list...", file=sys.stderr)
    networks = collect_networks(counts, decisions, args.base_prefix)

    subnet_count, extra_total, penalty_total = compute_metrics(
        networks, counts, args.penalty
    )
    print(
        f"\tfinal prefixes={subnet_count}, extra addresses={extra_total:,}, "
        f"penalty={penalty_total:,.2f}",
        file=sys.stderr,
    )

    with open(args.output, "w", encoding="ascii") as out:
        for network in networks:
            out.write(str(network))
            out.write("\n")

    print(f"Done. Subnets written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
