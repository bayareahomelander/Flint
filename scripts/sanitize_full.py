"""Runs GPT-2's forward pass at full size under NVIDIA compute-sanitizer.

Usage: python scripts/sanitize_full.py [--tools memcheck,racecheck,synccheck,initcheck]
       [--layers N] [--sequences 16] [--length 1024] [--allow-busy]

scripts/sanitize.py runs the test suite, whose GPT-2 tests shrink to a tiny model under the
tools, so the shapes training uses (16 sequences of 1,024 tokens, 12 layers) are never
checked there. This script checks them, one process per case:

    memcheck    12 layers, each buffer its own allocation, so its bounds are seen
    initcheck   12 layers, with separate buffers and as one allocation
    racecheck   2 layers at full width and length (the tool is slow, and a layer's
    synccheck   kernels are the same in every layer)

--layers overrides every case's layers, and --sequences and --length the batch, for a quick
run. Each case must report zero errors (racecheck zero warnings too) and a finite loss. It is
not part of precheck: it takes long, and the server's GPU is shared, so it refuses to start
while another process uses the GPU or holds its memory, unless --allow-busy is given (the
laptop's desktop always does; its 6 GiB hold the default cases' 5.4 GB only just, so use the
overrides there). Run it by hand after a change to the model's assembly. scripts/sanitize.py's
canaries, which precheck runs, show that the tools and the kernel filter see Flint's kernels.
Exits nonzero if a case fails.
"""

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import sanitize

ROOT = Path(__file__).resolve().parent.parent
FULL_LAYERS = 12
RACE_LAYERS = 2
# (layers, separate buffers) of each tool's cases.
CASES = {
    "memcheck": [(FULL_LAYERS, True)],
    "initcheck": [(FULL_LAYERS, True), (FULL_LAYERS, False)],
    "racecheck": [(RACE_LAYERS, True)],
    "synccheck": [(RACE_LAYERS, True)],
}
# As precheck.sh: more memory in use than this means someone else is on the GPU.
BUSY_MIB = 512
LOSS_LINE = "sanitize_full: mean loss "
# An untrained model's loss is near ln(50,257) = 10.8. This only shows that the pass ran to
# its end on finite values; the tests' gates check what it computes.
LOSS_RANGE = (9.0, 13.0)


def child(layers, sequences, length, separate):
    """One forward pass of GPT-2 with `layers` layers on random tokens, printing its loss."""
    sys.path.insert(0, str(ROOT / "tests"))
    import dataclasses

    import torch

    import flint_lib
    import gpt2_torch

    cfg = dataclasses.replace(gpt2_torch.GPT2_CONFIG, n_layer=layers)
    flint = flint_lib.Flint()
    config = flint_lib.Gpt2Config(
        layers=cfg.n_layer, heads=cfg.n_head, width=cfg.n_embd, block_size=cfg.block_size,
        vocab=cfg.vocab_size, padded_vocab=cfg.padded_vocab_size, sequences=sequences,
        length=length, separate_buffers=int(separate),
    )  # fmt: skip
    rows = sequences * length
    model = gpt2_torch.build_model(cfg, seed=0, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (rows + 1,), generator=generator, device="cuda")
    # The uint16 bit patterns Flint reads, in the int16 torch has.
    tokens = ((ids + 0x8000) % 0x10000 - 0x8000).to(torch.int16)
    with flint_lib.Gpt2(flint, config) as m:
        m.load(dict(model.named_parameters()))
        # Copies, since a slice from the second token on isn't 16-byte aligned.
        m.forward(tokens[:-1].clone(), tokens[1:].clone(), 2.0**16 / rows)
        loss = m.mean_loss()
    torch.cuda.synchronize()
    print(f"{LOSS_LINE}{loss!r}", flush=True)


def loss_of(output):
    """The loss a child printed, or None."""
    found = re.findall(re.escape(LOSS_LINE) + r"(\S+)", output)
    try:
        return float(found[-1]) if found else None
    except ValueError:
        return None


def loss_is_sane(loss):
    return loss is not None and math.isfinite(loss) and LOSS_RANGE[0] < loss < LOSS_RANGE[1]


def gpu_use():
    """(compute processes, MiB used) on GPU 0 as nvidia-smi reports them, or None if it can't
    tell. precheck.sh's test of a busy GPU: a process, or memory someone else holds."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout  # fmt: skip
        used = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout  # fmt: skip
        return sum(1 for line in apps.splitlines() if line.strip()), int(used.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tools", default=",".join(sanitize.TOOLS))
    parser.add_argument("--layers", type=int, help="override every case's layers")
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--separate", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        child(args.layers, args.sequences, args.length, bool(args.separate))
        return 0

    tools = args.tools.split(",")
    unknown = [tool for tool in tools if tool not in CASES]
    if unknown:
        parser.error(f"unknown tools {unknown}; choose from {sanitize.TOOLS}")
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        print("compute-sanitizer not found on PATH (enter the dev shell, or install the CUDA toolkit)")
        return 1
    if not args.allow_busy:
        use = gpu_use()
        if use is None or use[0] != 0 or use[1] > BUSY_MIB:
            seen = "nvidia-smi can't tell" if use is None else f"{use[0]} processes, {use[1]} MiB used"
            print(f"the GPU is busy or unknown ({seen}); try later, or pass --allow-busy")
            return 2

    failed = False
    script = str(Path(__file__).resolve())
    for tool in tools:
        for layers, separate in CASES[tool]:
            layers = args.layers or layers
            command = [sys.executable, script, "--child", "--layers", str(layers), "--sequences",
                       str(args.sequences), "--length", str(args.length), "--separate", str(int(separate))]  # fmt: skip
            code, output, seconds = sanitize.run(sanitize.sanitizer_command(sanitizer, tool, command))
            errors = sanitize.error_count(output)
            warnings = sanitize.warning_count(output) if tool == "racecheck" else 0
            passed = code == 0 and errors == 0 and warnings == 0 and loss_is_sane(loss_of(output))
            failed |= not passed
            name = (f"{tool} {layers} layers {args.sequences}x{args.length} "
                    f"{'separate' if separate else 'one allocation'}")  # fmt: skip
            sanitize.report(f"{name:<52}", passed, errors, seconds, output, warnings, tool)
            print(f"    loss {loss_of(output)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
