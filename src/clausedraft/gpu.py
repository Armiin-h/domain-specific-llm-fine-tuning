"""Report whether this machine can run a 4-bit QLoRA step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuReport:
    torch_version: str | None
    cuda_available: bool
    device_name: str | None
    total_vram_mib: int | None
    bitsandbytes_version: str | None
    four_bit_ok: bool
    detail: str


def format_report(report: GpuReport) -> str:
    vram = "-" if report.total_vram_mib is None else str(report.total_vram_mib)
    lines = [
        f"torch: {report.torch_version or 'not installed'}",
        f"cuda: {'yes' if report.cuda_available else 'no'}",
        f"device: {report.device_name or '-'}",
        f"vram_mib: {vram}",
        f"bitsandbytes: {report.bitsandbytes_version or 'not installed'}",
        f"four_bit: {'ok' if report.four_bit_ok else 'failed'}",
        f"detail: {report.detail}",
    ]
    return "\n".join(lines)


def probe_cuda() -> GpuReport:
    try:
        import torch
    except ImportError:
        return GpuReport(
            torch_version=None,
            cuda_available=False,
            device_name=None,
            total_vram_mib=None,
            bitsandbytes_version=None,
            four_bit_ok=False,
            detail="torch is not installed",
        )

    torch_version = torch.__version__
    bnb_version = _bitsandbytes_version()
    if not torch.cuda.is_available():
        return GpuReport(
            torch_version=torch_version,
            cuda_available=False,
            device_name=None,
            total_vram_mib=None,
            bitsandbytes_version=bnb_version,
            four_bit_ok=False,
            detail="CUDA is not available to this torch build",
        )

    device_name = torch.cuda.get_device_name(0)
    total_vram_mib = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    if bnb_version is None:
        return GpuReport(
            torch_version=torch_version,
            cuda_available=True,
            device_name=device_name,
            total_vram_mib=total_vram_mib,
            bitsandbytes_version=None,
            four_bit_ok=False,
            detail="bitsandbytes is not installed",
        )

    try:
        _run_four_bit_forward(torch)
    except Exception as exc:
        return GpuReport(
            torch_version=torch_version,
            cuda_available=True,
            device_name=device_name,
            total_vram_mib=total_vram_mib,
            bitsandbytes_version=bnb_version,
            four_bit_ok=False,
            detail=str(exc),
        )

    return GpuReport(
        torch_version=torch_version,
        cuda_available=True,
        device_name=device_name,
        total_vram_mib=total_vram_mib,
        bitsandbytes_version=bnb_version,
        four_bit_ok=True,
        detail="NF4 Linear4bit forward on cuda succeeded",
    )


def _bitsandbytes_version() -> str | None:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return None
    return getattr(bnb, "__version__", "unknown")


def _run_four_bit_forward(torch) -> None:
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).cuda()
    inputs = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    outputs = layer(inputs)
    if tuple(outputs.shape) != (2, 64):
        raise RuntimeError(f"unexpected output shape {tuple(outputs.shape)}")
    if not torch.isfinite(outputs).all():
        raise RuntimeError("4-bit matmul produced non-finite values")
