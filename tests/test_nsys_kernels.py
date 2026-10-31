"""scripts/nsys_kernels.py: kernel grouping from synthetic nsys CSV files."""

import importlib.util
import json

import pytest

import flint_lib

spec = importlib.util.spec_from_file_location(
    "nsys_kernels", flint_lib.ROOT / "scripts" / "nsys_kernels.py"
)
nsys_kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nsys_kernels)

HEADER = (
    '"Start (ns)","Duration (ns)","CorrId","GrdX","GrdY","GrdZ","BlkX","BlkY","BlkZ",'
    '"Reg/Trd","StcSMem (MB)","DymSMem (MB)","Bytes (MB)","Throughput (MB/s)",'
    '"SrcMemKd","DstMemKd","Device","Ctx","GreenCtx","Strm","Name"'
)
CUBLAS = "sm75_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x128x16"
FLINT = "void flint::gemm_nt_kernel<128, 128>(const float *, const float *, float *, int)"
BENCH = "void flint::microbench::mma_m16n8k8_f32(float, int, float4 *)"


def row(duration, grid, block, regs, static_mb, dynamic_mb, name):
    gx, gy, gz = grid
    bx, by, bz = block
    return (
        f'"1000","{duration}","7","{gx}","{gy}","{gz}","{bx}","{by}","{bz}","{regs}",'
        f'"{static_mb}","{dynamic_mb}","","","","","NVIDIA GeForce RTX 2080 Ti (0)","1","","7","{name}"'
    )


MEMCPY = '"900","500","6","","","","","","","","","","1.5","3000","Pageable","Device","NVIDIA (0)","1","","7","[CUDA memcpy Host-to-Device]"'


def synthetic(header=HEADER):
    return "\n".join(
        [
            "Processing events...",
            header,
            MEMCPY,
            row(100000, (8, 6, 1), (256, 1, 1), 128, "0.000000", "0.049152", CUBLAS),
            row(110000, (8, 6, 1), (256, 1, 1), 128, "0.000000", "0.049152", CUBLAS),
            row(120000, (8, 6, 1), (256, 1, 1), 128, "0.000000", "0.049152", CUBLAS),
            row(90000, (16, 12, 1), (128, 1, 1), 96, "0.000000", "0.032768", CUBLAS),
            row(80000, (8, 6, 1), (256, 1, 1), 254, "0.016384", "0.000000", FLINT),
            row(70000, (8, 6, 1), (256, 1, 1), 254, "0.016384", "0.000000", FLINT),
            row(5000, (1, 1, 1), (32, 1, 1), 32, "0.000000", "0.000000", BENCH),
            "",
        ]
    )


def by_name(groups):
    return {(g["name"], tuple(g["grid"])): g for g in groups}


def test_groups_sides_and_statistics(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(synthetic(), encoding="utf-8")
    result = nsys_kernels.build([path], "1024x768x768", command="flint_bench gemm")
    assert result["schema"] == "flint-nsys/1"
    assert result["case"] == "1024x768x768"
    groups = by_name(result["groups"])
    assert len(result["groups"]) == 4  # the memcpy row is skipped
    big = groups[(CUBLAS, (8, 6, 1))]
    assert big["side"] == "baseline"
    assert big["count"] == 3
    assert big["median_us"] == pytest.approx(110.0)
    assert big["min_us"] == pytest.approx(100.0)
    assert big["max_us"] == pytest.approx(120.0)
    assert big["threads_per_block"] == 256
    assert big["blocks"] == 48
    assert big["registers_per_thread"] == 128
    # Shared memory stays as reported, with the header's unit beside it.
    assert big["dynamic_smem"] == pytest.approx(0.049152)
    assert big["static_smem"] == 0
    assert result["smem_unit"] == {"static_smem": "mb", "dynamic_smem": "mb"}
    assert groups[(CUBLAS, (16, 12, 1))]["count"] == 1  # a second launch config
    flint = groups[(FLINT, (8, 6, 1))]
    assert flint["side"] == "flint"
    assert flint["count"] == 2
    assert flint["static_smem"] == pytest.approx(0.016384)
    assert groups[(BENCH, (1, 1, 1))]["side"] == "bench"
    assert [g["side"] for g in result["groups"]] == ["flint", "baseline", "baseline", "bench"]
    assert "git" not in result and "device" not in result


def test_bench_log_supplies_the_commit_and_gpu(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(synthetic(), encoding="utf-8")
    log = {
        "git": {"sha": "f1a308a", "dirty": False},
        "device": {"name": "NVIDIA GeForce RTX 2080 Ti", "compute_capability": "7.5",
                   "driver_version": "595", "sm_count": 68},
    }
    result = nsys_kernels.build([path], "c", bench_log=log, nsys_version="2026.3.2")
    assert result["git"] == {"sha": "f1a308a", "dirty": False}
    assert result["device"] == {"name": "NVIDIA GeForce RTX 2080 Ti",
                                "compute_capability": "7.5", "driver_version": "595"}
    assert result["nsys_version"] == "2026.3.2"


def test_files_with_different_units_refuse():
    other = HEADER.replace("StcSMem (MB)", "StcSMem (KB)")
    with pytest.raises(ValueError, match="static_smem"):
        nsys_kernels.kernels_from_text(synthetic(), synthetic(other))


def test_mangled_names():
    assert nsys_kernels.side_of("_ZN5flint11gemm_kernelEv") == "flint"
    assert nsys_kernels.side_of("_ZN5flint10microbench15mma_m16n8k8_f32EfiP6float4") == "bench"
    assert nsys_kernels.side_of("cutlass_80_tensorop_s1688gemm_128x128") == "baseline"


def test_renamed_alias_headers_parse(tmp_path):
    renamed = (
        HEADER.replace("GrdX", "Grid X").replace("BlkX", "Block X").replace("Duration (ns)", "Duration (us)")
    )
    renamed = renamed.replace("Reg/Trd", "Regs/Thread")
    path = tmp_path / "trace.csv"
    path.write_text(synthetic(renamed), encoding="utf-8")
    result = nsys_kernels.build([path], "x")
    big = by_name(result["groups"])[(CUBLAS, (8, 6, 1))]
    assert big["median_us"] == pytest.approx(110000.0)  # the duration is now in us
    assert big["registers_per_thread"] == 128


def test_missing_required_column_names_the_headers(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(synthetic(HEADER.replace('"GrdZ",', "")), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        nsys_kernels.build([path], "x")
    message = str(error.value)
    assert "grid_z" in message and "Duration (ns)" in message and "StcSMem (MB)" in message


def test_no_kernels_refuses(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("\n".join([HEADER, MEMCPY]), encoding="utf-8")
    with pytest.raises(ValueError, match="no kernel rows"):
        nsys_kernels.build([path], "x")


def test_scrub_check():
    nsys_kernels.scrub_check('{"command": "flint_bench gemm --out <out>"}')
    for leak in ("C:\\Temp\\x", "c:/temp", "/home/someone/x", "/Users/someone", "Users\\someone"):
        with pytest.raises(ValueError):
            nsys_kernels.scrub_check(leak)
    import getpass
    import socket

    with pytest.raises(ValueError):
        nsys_kernels.scrub_check(f"run by {getpass.getuser().upper()}")
    with pytest.raises(ValueError):
        nsys_kernels.scrub_check(f"on {socket.gethostname()}")


def test_command_line_refuses_a_path_in_the_command(tmp_path, monkeypatch):
    path = tmp_path / "trace.csv"
    path.write_text(synthetic(), encoding="utf-8")
    out = tmp_path / "kernels.json"
    argv = ["nsys_kernels.py", "--case", "c", "--out", str(out), "--command", "run /home/x/bench", str(path)]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        nsys_kernels.main()
    assert not out.exists()
    argv[argv.index("run /home/x/bench")] = "flint_bench gemm --out <out>"
    nsys_kernels.main()
    assert json.loads(out.read_text(encoding="utf-8"))["case"] == "c"
