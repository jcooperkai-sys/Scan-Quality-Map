import importlib
import sys

COMMANDS = {
    "map": ("sqm.map", "map scan quality over a region of an OME-Zarr volume"),
    "segment": ("sqm.segment", "score a tifxyz segment and add a quality.tif channel"),
    "pairs": ("sqm.pairs", "build voxel aligned pairs of the hazy and clean PHerc. Paris 4 scans"),
    "answer-key": ("sqm.answer_key", "check the features on the aligned pairs"),
    "resolvability": ("sqm.resolvability", "count real sheets a lower resolution scan still shows"),
    "fit": ("sqm.checkpoint2", "fit and evaluate the quality model on measured blocks"),
    "external": ("sqm.external", "apply the frozen model to a scroll it never saw"),
    "failures": ("sqm.failures", "compare surface predictions with hand labels per block"),
    "regions": ("sqm.regions", "compare segment quality inside marked boxes with the rest"),
}


def usage():
    lines = ["usage: sqm <command> [options]", "", "commands:"]
    lines += [f"  {name:14s} {text}" for name, (_, text) in COMMANDS.items()]
    lines += ["", "Run sqm <command> --help for the options of one command."]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(usage())
        sys.exit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
    name = sys.argv.pop(1)
    sys.argv[0] = f"sqm {name}"
    importlib.import_module(COMMANDS[name][0]).main()


if __name__ == "__main__":
    main()
