"""
Export the mesh from a Fluent HDF5 case file (.cas.h5) to a text-format
Fluent file that OpenFOAM's fluentMeshToFoam can read.

Fluent's solver mode cannot write a bare .msh file, so this writes a legacy
text case file (.cas) instead. The text case contains the same mesh sections
(nodes, faces, cells, zone names) as a .msh file, plus solver settings that
the converter ignores.

This is the only step that needs an ANSYS license; everything downstream is
pure OpenFOAM.

Usage:
  conda activate airway
  python export_fluent_mesh.py /path/to/case_file.cas.h5 [--output out.msh]

Output:
  - Text mesh: <case_stem>.msh (default, written next to this script)
  - Fluid properties: <case_stem>_materials.json (density, viscosity)
"""

import os
import sys
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import ansys.fluent.core as pyfluent


# ---------------------------------------------------------------------------
# Configuration (matches ../pyfluent_script/run_cfd_simulation.py)
# ---------------------------------------------------------------------------

LICENSE_SERVER = "1055@seas-gen-lic-01.gonzaga.edu"
LICENSE_INTERCONNECT = "2325@seas-gen-lic-01.gonzaga.edu"
FLUENT_PATH = "/usr/ansys_inc/v252/fluent/bin/fluent"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Fluent .cas.h5 mesh to text format for OpenFOAM."
    )
    parser.add_argument("case_file", help="Path to the Fluent case file (.cas.h5).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output mesh path (default: <case_stem>.msh in this directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    case_file = Path(args.case_file).resolve()
    if not case_file.exists():
        print(f"Error: Case file not found: {case_file}")
        sys.exit(1)

    stem = case_file.name.replace(".cas.h5", "").replace(".cas", "")
    output = Path(args.output).resolve() if args.output else Path(__file__).parent / f"{stem}.msh"

    print(f"Case file:   {case_file}")
    print(f"Output mesh: {output}\n")

    os.environ["ANSYSLMD_LICENSE_FILE"] = LICENSE_SERVER
    os.environ["ANSYSLI_SERVERS"] = LICENSE_INTERCONNECT
    os.environ["FLUENT_NO_AUTOMATIC_TRANSCRIPT"] = "1"

    print("Launching Fluent...")
    solver = pyfluent.launch_fluent(
        product_version="25.2",
        dimension=3,
        precision="double",
        processor_count=1,
        mode="solver",
        ui_mode="no_gui",
        fluent_path=FLUENT_PATH,
        start_transcript=False,
    )

    print(f"Reading case file: {case_file}")
    solver.settings.file.read_case(file_name=str(case_file))

    # Record fluid properties so the OpenFOAM case uses the same values
    fluids = {}
    for name in solver.settings.setup.materials.fluid.get_object_names():
        state = solver.settings.setup.materials.fluid[name].get_state()
        fluids[name] = {
            "density": state["density"],
            "viscosity": state["viscosity"],
        }
        print(f"  {name}: density={state['density']}, viscosity={state['viscosity']}")
    props_file = output.with_name(f"{stem}_materials.json")
    props_file.write_text(json.dumps(fluids, indent=2, default=str))
    print(f"Material properties written: {props_file}")

    # Legacy (non-HDF5) text output
    solver.settings.file.cff_files.set_state(False)
    solver.settings.file.binary_legacy_files.set_state(False)

    # Fluent appends .cas itself; write to a temp dir and then rename to .msh
    with tempfile.TemporaryDirectory() as tmp:
        tmp_case = Path(tmp) / f"{stem}.cas"
        print(f"Writing text case: {tmp_case}")
        solver.settings.file.write_case(file_name=str(tmp_case))
        solver.exit()

        written = tmp_case if tmp_case.exists() else next(Path(tmp).glob(f"{stem}*"))
        shutil.move(str(written), output)

    print(f"\nMesh exported: {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
