"""Hardware counters behind two of the milestone 1 models. Writes results/counters.json.

stride: global-load sectors per request and DRAM bytes per useful byte, against the 32-byte sector model.
bank:   shared-memory wavefronts per load instruction, against gcd(S, 32).

Needs GPU performance counters enabled (NVIDIA Control Panel, Developer, Manage GPU Performance Counters).
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXE = str(HERE / "bin" / ("anat.exe" if sys.platform == "win32" else "anat"))
NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe"


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def ncu(kernel, metrics, args):
    cmd = [NCU, "--kernel-name", f"regex:^{kernel}$", "--launch-skip", "1", "--launch-count", "1", "--metrics", ",".join(metrics),
           "--csv", "--page", "raw", EXE, *map(str, args), "--reps", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    rows = list(csv.DictReader(io.StringIO("\n".join(l for l in out.splitlines() if l.startswith('"')))))
    rows = [r for r in rows if num(r.get(metrics[0])) is not None]
    return {m: num(rows[0][m]) for m in metrics}


def main():
    res = {"stride": [], "bank": []}
    SM = ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
          "dram__bytes_read.sum", "lts__t_sectors_srcunit_tex_op_read.sum"]
    m = 1 << 21
    for s in (1, 2, 4, 8, 16, 32, 64):
        r = ncu("k_stride", SM, ["stride", "--s", s])
        row = {"s": s, "sectors_per_request": r[SM[1]] / r[SM[0]], "dram_bytes_per_useful_byte": r[SM[2]] / (4 * m),
               "l2_sectors_per_useful_sector": r[SM[3]] * 32 / (4 * m), "predicted_sectors_per_request": min(32, 4 * s)}
        res["stride"].append(row); print("stride", row, flush=True)
    BM = ["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "smsp__inst_executed_op_shared_ld.sum",
          "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]
    from math import gcd
    for s in (0, 1, 2, 3, 4, 8, 16, 31, 32, 33):
        r = ncu("k_bank", BM, ["bank", "--s", s])
        row = {"s": s, "wavefronts_per_inst": r[BM[0]] / r[BM[1]], "conflicts_per_inst": r[BM[2]] / r[BM[1]],
               "predicted_degree": 1 if s == 0 else gcd(s, 32)}
        res["bank"].append(row); print("bank", row, flush=True)
    (HERE / "results" / "counters.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
