# scripts/research_demo.py
import os
import json
import numpy as np
from src.io_utils.io_utils import ATFHandler
from src.analysis.action_potential import ActionPotentialProcessor
from src.research import ModelRegistry, compute_qc_metrics, detect_anomalies, generate_markdown_report


def run_demo(atf_path: str, save_report_path: str):
    # Betöltés
    h = ATFHandler(atf_path)
    h.load_atf()
    t = h.get_column('time')
    y = h.get_column('#1')  # első áramnyom

    # Processzor
    proc = ActionPotentialProcessor(y, t)
    proc.process_signal()

    # Jel kiválasztás (orange, ha van)
    if proc.orange_curve is not None:
        sig = {"time": proc.orange_curve_times if proc.orange_curve_times is not None else np.arange(len(proc.orange_curve))*1e-5,
               "current": proc.orange_curve}
    else:
        sig = {"time": t, "current": y}

    # QC + Anomália
    qc = compute_qc_metrics(sig)
    an = detect_anomalies(sig)

    # Modellek
    reg = ModelRegistry()
    reg.register("baseline_drift", lambda s: {"drift_pa_per_s": float(np.polyfit(s['time'], s['current'], 1)[0])})
    reg.register("spike_rate", lambda s: {"spike_rate_per_s": float(np.mean(s['current'] > 500.0) / (np.median(np.diff(s['time'])) + 1e-12))})
    models_out = reg.run_all(sig)

    # Integrálok (ha elérhető purple)
    integrals = {}
    try:
        if hasattr(proc, 'modified_hyperpol') and proc.modified_hyperpol is not None:
            # pC számítás időskálával
            hp = np.trapz(proc.modified_hyperpol, x=proc.modified_hyperpol_times*1000)
            dp = np.trapz(proc.modified_depol, x=proc.modified_depol_times*1000)
            cap_pF = abs(hp - dp) / abs(proc.params['V2'] - proc.params['V0'])
            integrals = {"hyperpol_pC": float(hp), "depol_pC": float(dp), "capacitance_nF": float(cap_pF/1000)}
    except Exception:
        pass

    # Riport
    ctx = {
        'file': os.path.basename(atf_path),
        'qc': qc,
        'anomaly': an,
        'models': models_out,
        'integrals': integrals,
    }
    md = generate_markdown_report(ctx)
    with open(save_report_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"OK: riport mentve: {save_report_path}")


if __name__ == "__main__":
    default_in = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '202304_0521.atf'))
    default_out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'research_report_demo.md'))
    run_demo(default_in, default_out)