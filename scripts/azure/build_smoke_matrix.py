#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def _issue_cluster_v1(args: argparse.Namespace) -> list[dict[str, str]]:
    lanes: list[dict[str, str]] = [
        {
            "lane_id": "win-py310",
            "os": "windows",
            "python": "3.10.11",
            "size": args.windows_size,
            "regions": args.windows_regions,
            "max_minutes": "75",
        },
        {
            "lane_id": "win-py311",
            "os": "windows",
            "python": "3.11.9",
            "size": args.windows_size,
            "regions": args.windows_regions,
            "max_minutes": "75",
        },
        {
            "lane_id": "win-py312",
            "os": "windows",
            "python": "3.12.7",
            "size": args.windows_size,
            "regions": args.windows_regions,
            "max_minutes": "75",
        },
        {
            "lane_id": "win-py313",
            "os": "windows",
            "python": "3.13.0",
            "size": args.windows_size,
            "regions": args.windows_regions,
            "max_minutes": "75",
        },
        {
            "lane_id": "linux-py310",
            "os": "linux",
            "python": args.linux_python,
            "size": args.linux_size,
            "regions": args.linux_regions,
            "max_minutes": "60",
        },
    ]
    return lanes


def _single(args: argparse.Namespace) -> list[dict[str, str]]:
    lanes: list[dict[str, str]] = []
    if args.run_windows:
        lanes.append(
            {
                "lane_id": "win-single",
                "os": "windows",
                "python": args.windows_python,
                "size": args.windows_size,
                "regions": args.windows_regions,
                "max_minutes": "75",
            }
        )
    if args.run_linux:
        lanes.append(
            {
                "lane_id": "linux-single",
                "os": "linux",
                "python": args.linux_python,
                "size": args.linux_size,
                "regions": args.linux_regions,
                "max_minutes": "60",
            }
        )
    if not lanes:
        raise SystemExit("single profile requested, but both windows and linux lanes are disabled")
    return lanes


def build_lanes(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.profile == "issue-cluster-v1":
        return _issue_cluster_v1(args)
    if args.profile == "single":
        return _single(args)
    raise SystemExit(f"Unsupported profile: {args.profile}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Azure GPU smoke matrix")
    parser.add_argument("--profile", required=True, choices=["issue-cluster-v1", "single"])
    parser.add_argument("--windows-size", default="Standard_NC4as_T4_v3")
    parser.add_argument("--linux-size", default="Standard_NC4as_T4_v3")
    parser.add_argument("--windows-regions", default="eastus,eastus2,westus2,westus")
    parser.add_argument("--linux-regions", default="eastus,eastus2,westus2,westus")
    parser.add_argument("--windows-python", default="3.11.9")
    parser.add_argument("--linux-python", default="3.10")
    parser.add_argument("--run-windows", action="store_true")
    parser.add_argument("--run-linux", action="store_true")
    parser.add_argument("--max-budget-usd", type=float, default=150.0)
    args = parser.parse_args()

    lanes = build_lanes(args)
    lane_budget = args.max_budget_usd / len(lanes)

    for lane in lanes:
        lane["lane_budget_usd"] = f"{lane_budget:.2f}"
        lane["profile"] = args.profile

    print(json.dumps({"include": lanes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
