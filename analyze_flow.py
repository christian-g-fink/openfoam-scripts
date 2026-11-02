"""
OpenFOAM flow analysis script for upper airway CFD simulation.
This is an analysis script, to be run after run_cfd_simulation.py.
OpenFOAM equivalent of ../pyfluent_script/analyze_flow.py: same metrics, same
CSV format, same sign convention (positive = flow entering the domain).

Reads the final solution of a solved OpenFOAM case and performs:
  1. Basic results: area-weighted avg static pressure at outlet,
     volume flow rates at leftinlet, rightinlet, outlet
  2. Choana analysis: create an axis-aligned plane (XY, YZ, or XZ) at the
     specified coordinate,
     compute pressure at choana, nasal resistance (total, left, right)

The surface integrals are computed by OpenFOAM's postProcess utility (run as a
separate process) with surfaceFieldValue function objects. The choana plane is
a cuttingPlane through the mesh cells, averaging the cell values the solver
computed (no interpolation to mesh points).

Usage:
  conda activate airway
  python analyze_flow.py /path/to/openfoam_case_dir --choana-z 0.083

  The choana plane is assumed to be perpendicular to the anterior-posterior direction,
  which may lie along any of the three coordinate axes. Specify exactly one of:
    --choana-x X: X-coordinate (m) of a YZ plane (anterior-posterior along x)
    --choana-y Y: Y-coordinate (m) of an XZ plane (anterior-posterior along y)
    --choana-z Z: Z-coordinate (m) of an XY plane (anterior-posterior along z)
  The coordinate should be determined from the geometry, established in ICEM.

  The case directory is the one written by run_cfd_simulation.py; its
  run_info.json supplies the air density used to convert pressures to Pa.

  Other optional flags:
    --compare FILE: Fluent results CSV from ../pyfluent_script/analyze_flow.py;
                    writes a side-by-side comparison CSV

Output: results printed to console and written to CSV (<case_dir>.csv, next to
the case directory), plus <case_dir>_vs_fluent.csv when --compare is given.
"""

import sys
import csv
import json
import argparse
from pathlib import Path

from foamlib import FoamCase, FoamFile

from run_cfd_simulation import load_openfoam_environment, INLETS, OUTLET


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Choana plane options: axis -> (plane normal, plane label)
CHOANA_PLANES = {
    "x": ([1, 0, 0], "YZ"),
    "y": ([0, 1, 0], "XZ"),
    "z": ([0, 0, 1], "XY"),
}

# postProcess dictionary written into the case's system/ directory
ANALYSIS_DICT = "analyzeFlowDict"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze flow results from upper airway CFD using OpenFOAM."
    )
    parser.add_argument(
        "case_dir",
        help="Path to the solved OpenFOAM case directory (from run_cfd_simulation.py).",
    )
    choana_group = parser.add_mutually_exclusive_group(required=True)
    for axis, (_, label) in CHOANA_PLANES.items():
        choana_group.add_argument(
            f"--choana-{axis}",
            type=float,
            help=f"{axis.upper()}-coordinate (in meters) for the choana "
            f"{label} plane surface.",
        )
    parser.add_argument(
        "--compare",
        default=None,
        help="Fluent results CSV to compare against (from analyze_flow.py).",
    )
    args = parser.parse_args()

    # Record which axis was specified for the choana plane
    for axis in CHOANA_PLANES:
        value = getattr(args, f"choana_{axis}")
        if value is not None:
            args.choana_axis = axis
            args.choana_coord = value
    return args


# ---------------------------------------------------------------------------
# Function objects
# ---------------------------------------------------------------------------

def patch_report(patch, operation, field):
    return {
        "type": "surfaceFieldValue",
        "libs": ["fieldFunctionObjects"],
        "regionType": "patch",
        "name": patch,
        "operation": operation,
        "fields": [field],
        "writeFields": False,
        "writeArea": True,
    }


def choana_report(point, normal, operation, field):
    # surfaceFieldValue always samples cell values on a sampled surface
    # (interpolationCell in OpenFOAM v2606), so no interpolate option is set
    return {
        "type": "surfaceFieldValue",
        "libs": ["fieldFunctionObjects"],
        "regionType": "sampledSurface",
        "name": "choana",
        "sampledSurfaceDict": {
            "type": "cuttingPlane",
            "planeType": "pointAndNormal",
            "pointAndNormalDict": {"point": point, "normal": normal},
        },
        "operation": operation,
        "fields": [field],
        "writeFields": False,
        "writeArea": True,
    }


def write_analysis_dict(case, args):
    normal, _ = CHOANA_PLANES[args.choana_axis]
    point = [0.0, 0.0, 0.0]
    point["xyz".index(args.choana_axis)] = args.choana_coord

    functions = {}
    for patch in INLETS + [OUTLET]:
        functions[f"analyze_flowRate_{patch}"] = patch_report(patch, "sum", "phi")
        functions[f"analyze_pAvg_{patch}"] = patch_report(patch, "areaAverage", "p")
    # The flow through the choana plane is a diagnostic: it should be close to
    # the outlet flow if the plane is a single complete cross-section
    functions["analyze_pAvg_choana"] = choana_report(point, normal, "areaAverage", "p")
    functions["analyze_flowRate_choana"] = choana_report(point, normal, "areaNormalIntegrate", "U")

    with FoamFile(case.path / "system" / ANALYSIS_DICT) as f:
        f["functions"] = functions
    return list(functions)


def read_report(case, time_name, function_name):
    """Return (value, area) at time_name from a surfaceFieldValue output file.

    Rows are: Time, Area, <operation>(<field>).
    """
    # postProcess names the output directory after the first time processed,
    # so search rather than assume
    # so search rather than assume. Use the last complete row for this time.
    found = None
    dats = sorted((case.path / "postProcessing" / function_name).glob("*/surfaceFieldValue.dat"))
    for dat in dats:
        for line in dat.read_text().splitlines():
            cols = line.split()
            if len(cols) == 3 and not line.startswith("#") and cols[0] == time_name:
                found = float(cols[2]), float(cols[1])
    if found is None:
        print(f"Error: no value for time {time_name} from function object {function_name}")
        sys.exit(1)
    return found


def latest_time(case):
    times = [t.name for t in case.path.glob("[0-9]*") if t.name != "0" and t.is_dir()]
    if not times:
        print(f"Error: no solution time directories in {case.path}")
        sys.exit(1)
    return max(times, key=float)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def write_comparison(results, fluent_csv, out_path):
    fluent = {}
    with open(fluent_csv, newline="") as f:
        for row in csv.DictReader(f):
            fluent[row["metric"]] = float(row["value"])

    print("\n" + "=" * 60)
    print("COMPARISON WITH FLUENT")
    print("=" * 60)
    print(f"  {'metric':40s} {'OpenFOAM':>12s} {'Fluent':>12s} {'diff %':>8s}")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "openfoam", "fluent", "diff_percent"])
        for key, val in results.items():
            ref = fluent.get(key)
            # Percent difference is meaningless for quantities that should be zero
            if ref is None or key == "vfr_net_cm3s" or ref == 0:
                diff = ""
            else:
                diff = 100 * (val - ref) / abs(ref)
            writer.writerow([key, val, "" if ref is None else ref, diff])
            ref_s = "" if ref is None else f"{ref:12.6f}"
            diff_s = "" if diff == "" else f"{diff:+8.2f}"
            print(f"  {key:40s} {val:12.6f} {ref_s:>12s} {diff_s:>8s}")
    print(f"\nComparison written to: {out_path}")


def main():
    args = parse_args()

    case_dir = Path(args.case_dir).resolve()
    run_info_file = case_dir / "run_info.json"
    if not run_info_file.exists():
        print(f"Error: {run_info_file} not found (is this a case from run_cfd_simulation.py?)")
        sys.exit(1)
    run_info = json.loads(run_info_file.read_text())
    density = run_info["density_kg_m3"]

    _, choana_label = CHOANA_PLANES[args.choana_axis]
    print(f"Case directory:      {case_dir}")
    print(f"Choana plane:        {choana_label} plane at "
          f"{args.choana_axis.upper()} = {args.choana_coord} m")
    print(f"Density:             {density} kg/m^3 (from run_info.json)")
    if not run_info.get("converged", False):
        print("WARNING: run_info.json says this solution did not converge.")
    print()

    # Dictionary to collect all results (same keys and order as the Fluent CSV)
    results = {}

    # -------------------------------------------------------------------
    # Compute surface integrals with OpenFOAM's postProcess
    # -------------------------------------------------------------------
    load_openfoam_environment()
    case = FoamCase(case_dir)
    time_name = latest_time(case)
    write_analysis_dict(case, args)
    print(f"Running postProcess on time {time_name}...")
    case.run(
        ["postProcess", "-dict", f"system/{ANALYSIS_DICT}", "-time", time_name,
         "-fields", "(p U phi)"],
        cpus=0,
    )
    print()

    def report(name):
        return read_report(case, time_name, name)

    # ===================================================================
    # PART 1: Basic Results
    # ===================================================================
    print("=" * 60)
    print("PART 1: Basic Results")
    print("=" * 60)

    # Area-weighted average static pressure at outlet (kinematic -> Pa)
    p_outlet = report(f"analyze_pAvg_{OUTLET}")[0] * density
    results["outlet_avg_static_pressure_Pa"] = p_outlet
    print(f"\n  Outlet area-weighted avg static pressure: {p_outlet:.6f} Pa")

    # Volume flow rates, cm^3/s. OpenFOAM's phi is positive leaving the domain;
    # negate to match Fluent (positive entering).
    vfr = {}
    for patch in INLETS + [OUTLET]:
        vfr[patch] = -report(f"analyze_flowRate_{patch}")[0] * 1e6
    vfr_left = vfr["leftinlet"]
    vfr_right = vfr["rightinlet"]
    vfr_outlet = vfr[OUTLET]
    results["vfr_leftinlet_cm3s"] = vfr_left
    results["vfr_rightinlet_cm3s"] = vfr_right
    results["vfr_outlet_cm3s"] = vfr_outlet
    vfr_net = vfr_left + vfr_right + vfr_outlet
    results["vfr_net_cm3s"] = vfr_net
    print(f"\n  Left inlet:  {vfr_left:.4f} cm^3/s")
    print(f"  Right inlet: {vfr_right:.4f} cm^3/s")
    print(f"  Outlet:      {vfr_outlet:.4f} cm^3/s")
    print(f"  Net (theoretically should be zero):         {vfr_net:.4f} cm^3/s")

    # ===================================================================
    # PART 2: Choana Analysis
    # ===================================================================
    print("\n" + "=" * 60)
    print("PART 2: Choana Analysis")
    print("=" * 60)

    p_choana_kin, choana_area = report("analyze_pAvg_choana")
    p_choana = p_choana_kin * density
    # Plane normal points toward +axis; report magnitude of flow through it.
    # Integrating cell-centre velocities over the cut is not exactly
    # conservative, so expect agreement with the outlet within a few percent.
    vfr_choana = abs(report("analyze_flowRate_choana")[0]) * 1e6
    results["choana_avg_static_pressure_Pa"] = p_choana
    print(f"\n  Choana area-weighted avg static pressure: {p_choana:.6f} Pa")
    print(f"  Choana plane area: {choana_area * 1e4:.4f} cm^2")
    print(f"  Flow through choana plane: {vfr_choana:.4f} cm^3/s "
          f"(should be within a few percent of the outlet flow; a large "
          f"difference means the plane is not a single complete cross-section)")

    # Nasal resistance calculations
    # Total nasal resistance = |P_choana| / Q_outlet (assumes pressure of 0 at infinity, even if inlet pressure is slightly negative)
    R_total = abs(p_choana / vfr_outlet)  # Pa / (cm^3/s) = Pa*s/cm^3
    results["total_nasal_resistance_Pa_s_per_cm3"] = R_total
    print(f"\n  Total nasal resistance: {R_total:.6f} Pa*s/cm^3")

    # Per-side inlet pressures (reported) and per-side resistances
    p_leftinlet = report("analyze_pAvg_leftinlet")[0] * density
    p_rightinlet = report("analyze_pAvg_rightinlet")[0] * density
    results["leftinlet_avg_static_pressure_Pa"] = p_leftinlet
    results["rightinlet_avg_static_pressure_Pa"] = p_rightinlet

    R_left = abs(p_choana / vfr_left)  # assumes pressure of 0 at infinity, even if inlet pressure is slightly negative
    R_right = abs(p_choana / vfr_right)
    results["left_nasal_resistance_Pa_s_per_cm3"] = R_left
    results["right_nasal_resistance_Pa_s_per_cm3"] = R_right
    print(f"\n  Left inlet pressure:  {p_leftinlet:.6f} Pa")
    print(f"  Right inlet pressure: {p_rightinlet:.6f} Pa")
    print(f"  Left nasal resistance:  {R_left:.6f} Pa*s/cm^3")
    print(f"  Right nasal resistance: {R_right:.6f} Pa*s/cm^3")

    # ===================================================================
    # Output Results
    # ===================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for key, val in results.items():
        print(f"  {key}: {val:.6f}")

    csv_path = case_dir.parent / f"{case_dir.name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, val in results.items():
            writer.writerow([key, val])
    print(f"\nResults written to: {csv_path}")

    # Diagnostics that are not part of the Fluent CSV format
    info = {
        "solution_time": time_name,
        "choana_axis": args.choana_axis,
        "choana_coordinate_m": args.choana_coord,
        "choana_area_cm2": choana_area * 1e4,
        "choana_flow_rate_cm3s": vfr_choana,
        "density_kg_m3": density,
    }
    info_path = case_dir / "analyze_flow_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"Diagnostics written to: {info_path}")

    if args.compare:
        fluent_csv = Path(args.compare).resolve()
        if not fluent_csv.exists():
            print(f"Error: Fluent CSV not found: {fluent_csv}")
            sys.exit(1)
        write_comparison(results, fluent_csv, case_dir.parent / f"{case_dir.name}_vs_fluent.csv")

    print("\nFlow analysis complete!")


if __name__ == "__main__":
    main()
