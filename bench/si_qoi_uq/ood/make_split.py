"""Leave-types-out split for the out-of-distribution Mondrian check (paper 3.2): GAP-18's cracks and screw dislocations
are removed from training, calibration and the in-distribution test file, and form the OOD set.  Frames are copied
verbatim (text level), so labels and keys are bit-identical to the source files.

    python -m bench.si_qoi_uq.ood.make_split   -> $OOD_DATA/{train,test,ood}.xyz, split.json
"""
import json, os, pathlib, re

SRC = pathlib.Path("/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost")
OUT = pathlib.Path(os.environ.get("OOD_DATA", "/storage/eng/essswb/projects/si-qoi-uq-paper/data/ood_holdout"))
HELD = ("crack_111_1-10", "crack_110_1-10", "screw_disloc")


def frames(path):
    lines = open(path).read().splitlines(keepends=True)
    i = 0
    while i < len(lines):
        n = int(lines[i])
        yield "".join(lines[i:i + n + 2]), re.search(r"config_type=(\S+)", lines[i + 1]).group(1)
        i += n + 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    count = {}
    with open(OUT / "ood.xyz", "w") as ood:
        for split in ("train", "test"):
            with open(OUT / f"{split}.xyz", "w") as keep:
                for fr, t in frames(SRC / f"{split}.xyz"):
                    held = t in HELD
                    (ood if held else keep).write(fr)
                    k = f"{split}/{'ood' if held else 'kept'}/{t if held else 'other'}"
                    count[k] = count.get(k, 0) + 1
    json.dump({"source": str(SRC), "held_types": HELD, "counts": count}, open(OUT / "split.json", "w"), indent=1)
    print(json.dumps(count, indent=1))


if __name__ == "__main__":
    main()
