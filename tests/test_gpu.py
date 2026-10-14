from clausedraft.gpu import GpuReport, format_report, probe_cuda


def test_format_report_lists_each_field():
    report = GpuReport(
        torch_version="2.14.1+cu130",
        cuda_available=True,
        device_name="NVIDIA GeForce RTX 5060 Laptop GPU",
        total_vram_mib=8151,
        bitsandbytes_version="0.50.2",
        four_bit_ok=True,
        detail="NF4 Linear4bit forward on cuda succeeded",
    )
    text = format_report(report)
    assert "torch: 2.14.1+cu130" in text
    assert "cuda: yes" in text
    assert "vram_mib: 8151" in text
    assert "four_bit: ok" in text


def test_probe_returns_a_report_without_raising():
    report = probe_cuda()
    assert isinstance(report, GpuReport)
    assert isinstance(report.detail, str) and report.detail
    if not report.cuda_available:
        assert report.four_bit_ok is False
