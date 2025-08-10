# src/research/reporting.py
from typing import Dict, Any


def _fmt(v, digits=3):
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def generate_markdown_report(context: Dict[str, Any]) -> str:
    """
    Markdown riport a kutatási metrikákról.
    context tartalma (példa):
      {
        'file': 'path',
        'qc': {...},
        'anomaly': {...},
        'models': { name: {...}},
        'integrals': { 'hyperpol_pC': float, 'depol_pC': float, 'capacitance_nF': float}
      }
    """
    lines = []
    lines.append(f"## Jelentés: {context.get('file','Ismeretlen fájl')}")
    lines.append("")

    if 'qc' in context:
        qc = context['qc']
        lines.append("### QC metrikák")
        lines.append(f"- SNR: {_fmt(qc.get('snr'))}")
        lines.append(f"- Drift (pA/s): {_fmt(qc.get('drift_pa_per_s'))}")
        lines.append(f"- Outlier arány: {_fmt(qc.get('outlier_ratio'))}")
        lines.append(f"- Mintavételi jitter: {_fmt(qc.get('sampling_jitter'))}")
        lines.append("")

    if 'integrals' in context:
        integ = context['integrals']
        lines.append("### Integrálok és kapacitás")
        lines.append(f"- Hyperpol (pC): {_fmt(integ.get('hyperpol_pC'))}")
        lines.append(f"- Depol (pC): {_fmt(integ.get('depol_pC'))}")
        lines.append(f"- Kapacitás (nF): {_fmt(integ.get('capacitance_nF'))}")
        lines.append("")

    if 'anomaly' in context:
        an = context['anomaly']
        lines.append("### Anomáliák")
        lines.append(f"- Detektált pontok: {an.get('count',0)}")
        lines.append(f"- Küszöb átlag: {_fmt(an.get('threshold_mean'))}")
        lines.append(f"- Reziduum átlag: {_fmt(an.get('residual_mean'))}")
        lines.append("")

    if 'models' in context and isinstance(context['models'], dict):
        lines.append("### Modell kimenetek")
        for name, out in context['models'].items():
            lines.append(f"- {name}:")
            if isinstance(out, dict):
                for k, v in out.items():
                    lines.append(f"  - {k}: {_fmt(v)}")
            else:
                lines.append(f"  - output: {out}")
        lines.append("")

    return "\n".join(lines)