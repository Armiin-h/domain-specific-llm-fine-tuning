"""Print a CUDA / 4-bit readiness report. Exit 0 only when the probe passes."""

from clausedraft.gpu import format_report, probe_cuda


def main() -> int:
    report = probe_cuda()
    print(format_report(report))
    if report.four_bit_ok:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
