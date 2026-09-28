"""
Entry point:  python -m geometry_opt.run_pipeline [--models v0218 v140] [--materials PLA PETG] [--no-cache]
                                                   [--symmetric]

For each trained policy: record its gait (cached), validate the inverse dynamics against the
forward simulation, then run the dynamics / actuator / structure loop for each material.
Writes geometry_opt/results_summary.md, per-run time series (results/timeseries_*.npz) and CSVs.
With --symmetric (same motor left/right) the outputs go to results/symmetric_motors/ and
results_summary_symmetric.md instead, so both variants coexist.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pandas as pd

from . import config as C
from . import catalog as CAT
from . import thesis_params as TP
from . import simulation as SIM
from . import inverse_dynamics as ID
from . import report as R
from .model_builder import Design, build_model, mass_breakdown, segment_lengths, JOINT_NAMES
from .pipeline import optimize, save_timeseries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(C.MODELS))
    ap.add_argument("--materials", nargs="+", default=list(C.MATERIALS))
    ap.add_argument("--no-cache", action="store_true", help="re-record the gaits")
    ap.add_argument("--symmetric", action="store_true", help="same motor on the left and right leg")
    args = ap.parse_args(argv)
    if args.symmetric:
        C.SYMMETRIC_MOTORS = True
    out = C.output_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"== Motors left/right: {'same (symmetric)' if C.SYMMETRIC_MOTORS else 'independent'} -> {out}")
    t0 = time.time()

    print("== Thesis parameters")
    thesis = TP.load()
    ok, check = TP.check_against_model(thesis)
    print("\n".join(check))
    catalog = CAT.load()
    print(f"== Actuator catalog: {len(catalog)} motors")
    lengths = segment_lengths()

    md = [R.header(thesis, check)]
    all_results = {}
    for key in args.models:
        print(f"\n== {C.MODELS[key]['label']}")
        gait = SIM.record_gait(key, use_cache=not args.no_cache)
        print(f"  gait: {gait['qpos'].shape[0]} states, {float(gait['speed']):.3f} m/s")
        m0 = build_model(Design())
        val = ID.validate(ID.run(m0, gait), gait, mass_breakdown(m0, Design())["total"])
        print(f"  ID validation: max torque RMS error {val['tau_rms_err'].max():.3f} N·m, "
              f"hinge check {val['hinge_check_Nm']:.3f} N·m")
        results = []
        for mat in args.materials:
            print(f"  -- {mat}")
            res = optimize(gait, mat, catalog, lengths)
            ts = save_timeseries(res, key, float(gait["dt"]), int(gait["n_envs"]), int(gait["n_per_env"]))
            pd.DataFrame(res["selection"]).to_csv(out / f"motors_{key}_{mat}.csv", index=False)
            pd.DataFrame(res["structure"]).T.to_csv(out / f"structure_{key}_{mat}.csv")
            pd.DataFrame([dict(iter=h["iter"], total=h["mass"]["total"], torso=h["mass"]["torso"],
                               motors=h["mass"]["motors"], structure=h["mass"]["structure"],
                               next_total=h["next_mass"]["total"], dm=h["dm"], motors_changed=not h["motors_same"],
                               base_residual_max=h["base_residual_max"], selection="|".join(h["selection"]),
                               **{f"tau_max_{j}": v for j, v in zip(JOINT_NAMES, h["tau_max"])},
                               **{f"tau_rms_{j}": v for j, v in zip(JOINT_NAMES, h["tau_rms"])})
                          for h in res["history"]]).to_csv(out / f"history_{key}_{mat}.csv", index=False)
            print(f"     {'converged' if res['converged'] else 'NOT converged'} in {res['iterations']} iterations;"
                  f" final mass {res['final_mass']['total']:.2f} kg; time series -> {ts.name}")
            results.append(res)
        all_results[key] = results
        md.append(R.model_section(key, gait, val, results))

    md.insert(1, R.comparison(all_results))
    md.append(R.notes())
    text = "\n".join(md)
    C.summary_path().write_text(text)
    print("\n" + "=" * 100 + "\n" + text)
    print(f"\nWritten {C.summary_path()}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
