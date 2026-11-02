"""
OpenFOAM script for upper airway CFD simulation.

OpenFOAM equivalent of ../pyfluent_script/run_cfd_simulation.py: steady-state
laminar airflow through an upper airway geometry, on the same mesh that
Fluent uses (exported with export_fluent_mesh.py).

Simulation parameters (Fluent equivalent in parentheses):
  - Steady-state, incompressible, laminar: simpleFoam (Fluent: steady, laminar)
  - Pressure-velocity coupling: SIMPLEC (Fluent: Coupled)
  - Inlets (leftinlet, rightinlet): totalPressure p0 = 0
    (Fluent: pressure-inlet, gauge total pressure = 0)
  - Outlet: flowRateOutletVelocity, volumetric flow = mass flow / density
    (Fluent: mass-flow-outlet, 0.0003 kg/s)
  - Walls (nasopharynx, leftnasalcavity, rightnasalcavity): no-slip
  - Second-order spatial discretization (linearUpwind for velocity)
  - Runs until all initial residuals fall below 1e-4, or until the maximum
    number of iterations is reached. SIMPLE-type solvers need more iterations
    than Fluent's coupled solver, hence the larger default.
  - Fluid properties come from the Fluent case (<mesh_stem>_materials.json).

OpenFOAM's programs are run as separate processes (via foamlib); nothing in
this script links against OpenFOAM libraries.

Usage:
  conda activate airway
  python run_cfd_simulation.py /path/to/mesh.msh

  mesh.msh is the text-format Fluent mesh written by export_fluent_mesh.py.

  Optional flags:
    --materials FILE: fluid properties JSON (default: <mesh_stem>_materials.json)
    --case-dir DIR: OpenFOAM case directory to create (default: <mesh_stem>_solved_flow)
    --processors N: number of parallel solver processes (default: 6)
    --iterations N: maximum number of solver iterations (default: 2000)
    --mass-flow-rate M: outlet mass flow rate in kg/s (default: 0.0003)
    --convergence C: residual convergence criterion (default: 1e-4)
    --overwrite: delete the case directory first if it already exists

Output (in the case directory):
    - Solved OpenFOAM case (mesh, final fields, log.* files)
    - postProcessing/: per-iteration flow rates and average pressures at the
      inlets and outlet
    - run_info.json: OpenFOAM version, settings, convergence status
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import foamlib
from foamlib import FoamCase, FoamFile, FoamFieldFile


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OpenFOAM installation (ESI v2606, pinned with apt-mark hold)
OPENFOAM_BASHRC = "/usr/lib/openfoam/openfoam2606/etc/bashrc"
OPENFOAM_PACKAGE = "openfoam2606"

# Simulation parameters
NUM_PROCESSORS = 6  # physical cores; Open MPI will not oversubscribe the 12 hardware threads
NUM_ITERATIONS = 2000
MASS_FLOW_RATE = 0.0003  # kg/s (250 cm^3/s at the case's air density of 1.2 kg/m^3)
CONVERGENCE_CRITERIA = 1e-4

# Patch names from the Fluent mesh
INLETS = ["leftinlet", "rightinlet"]
OUTLET = "outlet"
WALLS = ["nasopharynx", "leftnasalcavity", "rightnasalcavity"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run upper airway CFD simulation using OpenFOAM."
    )
    parser.add_argument(
        "mesh_file",
        help="Path to the text-format Fluent mesh (.msh) from export_fluent_mesh.py.",
    )
    parser.add_argument(
        "--materials",
        default=None,
        help="Fluid properties JSON (default: <mesh_stem>_materials.json).",
    )
    parser.add_argument(
        "--case-dir",
        default=None,
        help="OpenFOAM case directory to create (default: <mesh_stem>_solved_flow).",
    )
    parser.add_argument(
        "--processors",
        type=int,
        default=NUM_PROCESSORS,
        help=f"Number of parallel solver processes (default: {NUM_PROCESSORS}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Maximum number of solver iterations (default: {NUM_ITERATIONS}).",
    )
    parser.add_argument(
        "--mass-flow-rate",
        type=float,
        default=MASS_FLOW_RATE,
        help=f"Outlet mass flow rate in kg/s (default: {MASS_FLOW_RATE}).",
    )
    parser.add_argument(
        "--convergence",
        type=float,
        default=CONVERGENCE_CRITERIA,
        help=f"Residual convergence criterion (default: {CONVERGENCE_CRITERIA}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Delete the case directory first if it already exists.",
    )
    return parser.parse_args()


def load_openfoam_environment():
    """Source the OpenFOAM bashrc and copy its environment into this process."""
    if not Path(OPENFOAM_BASHRC).exists():
        print(f"Error: OpenFOAM not found at {OPENFOAM_BASHRC}")
        sys.exit(1)
    out = subprocess.run(
        ["bash", "-c", f"source {OPENFOAM_BASHRC} >/dev/null 2>&1; env -0"],
        capture_output=True,
        check=True,
    ).stdout
    for entry in out.split(b"\0"):
        key, sep, value = entry.decode().partition("=")
        if sep:
            os.environ[key] = value
    if "WM_PROJECT_DIR" not in os.environ:
        print("Error: sourcing the OpenFOAM environment failed.")
        sys.exit(1)


def openfoam_package_version():
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Version}", OPENFOAM_PACKAGE],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_materials(path):
    """Read air density and viscosity written by export_fluent_mesh.py."""
    fluids = json.loads(Path(path).read_text())
    if len(fluids) != 1:
        print(f"Error: expected one fluid in {path}, found {list(fluids)}")
        sys.exit(1)
    name, props = next(iter(fluids.items()))
    for key in ["density", "viscosity"]:
        if props[key]["option"] != "value":
            print(f"Error: {name} {key} is not a constant ({props[key]['option']}).")
            sys.exit(1)
    return name, props["density"]["value"], props["viscosity"]["value"]


# ---------------------------------------------------------------------------
# Case dictionaries
# ---------------------------------------------------------------------------

def surface_report(patch, operation, field):
    """Function object that logs a patch integral every iteration."""
    return {
        "type": "surfaceFieldValue",
        "libs": ["fieldFunctionObjects"],
        "regionType": "patch",
        "name": patch,
        "operation": operation,
        "fields": [field],
        "writeFields": False,
        "log": False,
        "writeControl": "timeStep",
        "writeInterval": 1,
    }


def write_control_dict(case, args):
    functions = {
        "solverInfo": {
            "type": "solverInfo",
            "libs": ["utilityFunctionObjects"],
            "fields": ["U", "p"],
            "writeResidualFields": False,
        },
    }
    # Volume flow rates (m^3/s) and area-weighted average kinematic pressure
    # (m^2/s^2) at the inlets and outlet, recorded every iteration
    for patch in INLETS + [OUTLET]:
        functions[f"flowRate_{patch}"] = surface_report(patch, "sum", "phi")
        functions[f"pAvg_{patch}"] = surface_report(patch, "areaAverage", "p")

    with FoamFile(case.path / "system" / "controlDict") as f:
        f["application"] = "simpleFoam"
        f["startFrom"] = "startTime"
        f["startTime"] = 0
        f["stopAt"] = "endTime"
        f["endTime"] = args.iterations
        f["deltaT"] = 1
        # Write only the final result (simpleFoam also writes when it converges)
        f["writeControl"] = "timeStep"
        f["writeInterval"] = args.iterations
        f["purgeWrite"] = 0
        f["writeFormat"] = "ascii"
        f["writePrecision"] = 12
        f["writeCompression"] = False
        f["timeFormat"] = "general"
        f["timePrecision"] = 8
        f["runTimeModifiable"] = False
        f["functions"] = functions


def write_fv_schemes(case):
    # Limited gradient and non-orthogonal corrections handle the poorer cells
    # of the converted Fluent polyhedral mesh (max non-orthogonality ~80 deg).
    with FoamFile(case.path / "system" / "fvSchemes") as f:
        f["ddtSchemes"] = {"default": "steadyState"}
        f["gradSchemes"] = {
            "default": ("Gauss", "linear"),
            "grad(U)": ("cellLimited", "Gauss", "linear", 1),
        }
        f["divSchemes"] = {
            "default": "none",
            "div(phi,U)": ("bounded", "Gauss", "linearUpwind", "grad(U)"),
            "div((nuEff*dev2(T(grad(U)))))": ("Gauss", "linear"),
        }
        f["laplacianSchemes"] = {"default": ("Gauss", "linear", "limited", "corrected", 0.333)}
        f["interpolationSchemes"] = {"default": "linear"}
        f["snGradSchemes"] = {"default": ("limited", "corrected", 0.333)}


def write_fv_solution(case, args):
    with FoamFile(case.path / "system" / "fvSolution") as f:
        f["solvers"] = {
            "p": {
                "solver": "GAMG",
                "smoother": "GaussSeidel",
                "tolerance": 1e-8,
                "relTol": 0.05,
            },
            "U": {
                "solver": "smoothSolver",
                "smoother": "symGaussSeidel",
                "tolerance": 1e-9,
                "relTol": 0.1,
            },
        }
        f["SIMPLE"] = {
            "consistent": True,
            "nNonOrthogonalCorrectors": 2,
            "residualControl": {"p": args.convergence, "U": args.convergence},
        }
        f["relaxationFactors"] = {
            "equations": {"U": 0.7},
            "fields": {"p": 0.7},
        }


def write_decompose_par_dict(case, args):
    with FoamFile(case.path / "system" / "decomposeParDict") as f:
        f["numberOfSubdomains"] = args.processors
        f["method"] = "scotch"


def write_constant_dicts(case, nu):
    with FoamFile(case.path / "constant" / "transportProperties") as f:
        f["transportModel"] = "Newtonian"
        f["nu"] = nu
    with FoamFile(case.path / "constant" / "turbulenceProperties") as f:
        f["simulationType"] = "laminar"


def write_initial_fields(case, volumetric_flow_rate):
    walls = "(" + "|".join(WALLS) + ")"
    inlets = "(" + "|".join(INLETS) + ")"

    with FoamFieldFile(case.path / "0" / "U") as f:
        f.class_ = "volVectorField"
        f.object_ = "U"
        f.dimensions = FoamFile.DimensionSet(length=1, time=-1)
        f.internal_field = [0.0, 0.0, 0.0]
        f.boundary_field = {
            f'"{inlets}"': {
                "type": "pressureInletOutletVelocity",
                "value": [0.0, 0.0, 0.0],
            },
            OUTLET: {
                "type": "flowRateOutletVelocity",
                "volumetricFlowRate": volumetric_flow_rate,
                "value": [0.0, 0.0, 0.0],
            },
            f'"{walls}"': {"type": "noSlip"},
        }

    # Kinematic pressure (p/rho); gauge, relative to atmosphere
    with FoamFieldFile(case.path / "0" / "p") as f:
        f.class_ = "volScalarField"
        f.object_ = "p"
        f.dimensions = FoamFile.DimensionSet(length=2, time=-2)
        f.internal_field = 0.0
        f.boundary_field = {
            f'"{inlets}"': {
                "type": "totalPressure",
                "p0": ("uniform", 0.0),
                "value": 0.0,
            },
            OUTLET: {"type": "zeroGradient"},
            f'"{walls}"': {"type": "zeroGradient"},
        }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def solver_summary(case):
    """Return (converged, iterations) from the simpleFoam log."""
    log = (case.path / "log.simpleFoam").read_text()
    converged = "SIMPLE solution converged" in log
    times = [float(t.name) for t in case.path.glob("[0-9]*") if t.name != "0"]
    iterations = int(max(times)) if times else 0
    return converged, iterations


def last_value(case, function_name):
    """Last recorded value of a surfaceFieldValue function object."""
    dat = case.path / "postProcessing" / function_name / "0" / "surfaceFieldValue.dat"
    lines = [ln for ln in dat.read_text().splitlines() if ln and not ln.startswith("#")]
    return float(lines[-1].split()[1])


def main():
    args = parse_args()

    mesh_file = Path(args.mesh_file).resolve()
    if not mesh_file.exists():
        print(f"Error: Mesh file not found: {mesh_file}")
        sys.exit(1)

    stem = mesh_file.name.removesuffix(".msh")
    materials_file = (
        Path(args.materials).resolve()
        if args.materials
        else mesh_file.with_name(f"{stem}_materials.json")
    )
    if not materials_file.exists():
        print(f"Error: Materials file not found: {materials_file}")
        sys.exit(1)

    case_dir = (
        Path(args.case_dir).resolve()
        if args.case_dir
        else mesh_file.with_name(f"{stem}_solved_flow")
    )
    if case_dir.exists():
        if not args.overwrite:
            print(f"Error: Case directory already exists: {case_dir} (use --overwrite)")
            sys.exit(1)
        shutil.rmtree(case_dir)

    fluid, density, viscosity = load_materials(materials_file)
    nu = viscosity / density
    volumetric_flow_rate = args.mass_flow_rate / density

    print(f"Mesh file:    {mesh_file}")
    print(f"Materials:    {materials_file}")
    print(f"Case dir:     {case_dir}")
    print(f"Processors:   {args.processors}")
    print(f"Iterations:   {args.iterations} (maximum)")
    print(f"Mass flow:    {args.mass_flow_rate} kg/s "
          f"= {volumetric_flow_rate * 1e6:.4g} cm^3/s")
    print(f"Fluid:        {fluid}, density {density} kg/m^3, "
          f"viscosity {viscosity} Pa s (nu = {nu:.6g} m^2/s)")
    print()

    # -----------------------------------------------------------------------
    # 1. OpenFOAM environment
    # -----------------------------------------------------------------------
    load_openfoam_environment()
    of_version = os.environ.get("WM_PROJECT_VERSION", "unknown")
    of_package = openfoam_package_version()
    print(f"OpenFOAM {of_version} (package {of_package}), foamlib {foamlib.__version__}\n")

    # -----------------------------------------------------------------------
    # 2. Create the case and write its dictionaries
    # -----------------------------------------------------------------------
    print("Writing case dictionaries...")
    for sub in ["0", "constant", "system"]:
        (case_dir / sub).mkdir(parents=True)
    case = FoamCase(case_dir)
    write_control_dict(case, args)
    write_fv_schemes(case)
    write_fv_solution(case, args)
    write_decompose_par_dict(case, args)
    write_constant_dicts(case, nu)
    write_initial_fields(case, volumetric_flow_rate)

    # -----------------------------------------------------------------------
    # 3. Convert the Fluent mesh (polyhedral, so fluent3DMeshToFoam)
    # -----------------------------------------------------------------------
    print("Converting mesh (fluent3DMeshToFoam)...")
    case.run(["fluent3DMeshToFoam", str(mesh_file)], cpus=0)
    case.run(["checkMesh"], cpus=0, check=False)

    # -----------------------------------------------------------------------
    # 4. Solve
    # -----------------------------------------------------------------------
    started = datetime.now()
    if args.processors > 1:
        print(f"Decomposing mesh into {args.processors} subdomains...")
        case.decompose_par()
        print(f"Running simpleFoam on {args.processors} processors...")
        case.run(["simpleFoam"], parallel=True, cpus=args.processors)
        print("Reconstructing...")
        case.reconstruct_par()
        for proc_dir in case_dir.glob("processor*"):
            shutil.rmtree(proc_dir)
    else:
        print("Running simpleFoam...")
        case.run(["simpleFoam"], parallel=False, cpus=1)
    finished = datetime.now()

    converged, iterations = solver_summary(case)
    status = "converged" if converged else "did NOT converge"
    print(f"simpleFoam {status} after {iterations} iterations "
          f"({(finished - started).total_seconds():.0f} s).\n")

    # -----------------------------------------------------------------------
    # 5. Verify volume flow rates (positive = leaving the domain)
    # -----------------------------------------------------------------------
    print("Volume flow rates (cm^3/s, negative = entering the domain):")
    for patch in INLETS + [OUTLET]:
        print(f"  {patch:12s} {last_value(case, f'flowRate_{patch}') * 1e6:10.4f}")
    print(f"Outlet average static pressure: "
          f"{last_value(case, f'pAvg_{OUTLET}') * density:.4f} Pa\n")
    # Note: OpenFOAM's sign convention is the opposite of Fluent's, where the
    # outlet reads ~-250 cm^3/s.

    # -----------------------------------------------------------------------
    # 6. Record run information
    # -----------------------------------------------------------------------
    run_info = {
        "openfoam_version": of_version,
        "openfoam_package": of_package,
        "foamlib_version": foamlib.__version__,
        "python_version": platform.python_version(),
        "host": platform.node(),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "mesh_file": str(mesh_file),
        "mesh_sha256": sha256(mesh_file),
        "materials_file": str(materials_file),
        "fluid": fluid,
        "density_kg_m3": density,
        "viscosity_Pa_s": viscosity,
        "nu_m2_s": nu,
        "mass_flow_rate_kg_s": args.mass_flow_rate,
        "volumetric_flow_rate_m3_s": volumetric_flow_rate,
        "processors": args.processors,
        "max_iterations": args.iterations,
        "convergence_criterion": args.convergence,
        "converged": converged,
        "iterations": iterations,
    }
    (case_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Run information written: {case_dir / 'run_info.json'}")
    print("\nSimulation complete!" if converged else
          "\nSimulation finished without converging; check log.simpleFoam.")


if __name__ == "__main__":
    main()
