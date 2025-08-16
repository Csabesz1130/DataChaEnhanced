import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st

# Lazy pandas import
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# ai_monitor működik tiszta stdlibbel
from src.ai_excel_learning.ai_monitor import (
    get_ai_monitor,
    MetricType,
)


# --- Helpers & cached singletons ---
@st.cache_resource
def get_models_manager(models_dir: str = "models"):
    try:
        from src.ai_excel_learning.ml_models import ExcelMLModels
        return ExcelMLModels(models_dir=models_dir)
    except Exception as e:
        st.session_state["models_import_error"] = str(e)
        return None


@st.cache_resource
def get_model_version_manager():
    try:
        from src.ai_excel_learning.model_manager import ModelManager
        return ModelManager(models_dir="models", versions_dir="model_versions", deployments_dir="deployments")
    except Exception as e:
        st.session_state["versions_import_error"] = str(e)
        return None


@st.cache_resource
def get_adaptive_system():
    try:
        from src.ai_excel_learning.adaptive_learning import AdaptiveLearningRate
        return AdaptiveLearningRate()
    except Exception as e:
        st.session_state["adaptive_import_error"] = str(e)
        return None


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def deps_warning_box():
    missing = []
    if pd is None:
        missing.append("pandas")
    if "models_import_error" in st.session_state:
        missing.append("ml_models (TensorFlow/scikit-learn függőségek)")
    if "versions_import_error" in st.session_state:
        missing.append("model_manager (pandas/joblib függőségek)")
    if "adaptive_import_error" in st.session_state:
        missing.append("adaptive_learning (numpy/pandas)")
    if missing:
        with st.expander("Hiányzó függőségek / környezeti megjegyzések", expanded=False):
            st.warning("\n".join([f"- {m}" for m in missing]))
            st.markdown(
                """
                Javasolt gyorsindítás:
                
                - Használj Python 3.11 környezetet (ajánlott a TensorFlow kompatibilitás miatt)
                - Telepítés: `pip install -r requirements.txt`
                - Ha nem kell TensorFlow, próbáld: `pip install streamlit pandas numpy joblib scikit-learn`
                
                Streamlit indítás:
                
                ```bash
                streamlit run ai_lab_app.py --server.port 8501 --server.address 0.0.0.0
                ```
                """
            )


# --- UI Sections ---

def ui_training(models, versions):
    st.subheader("Tanítás (CSV gyors betanítás)")

    if pd is None:
        st.info("A tanításhoz szükséges a pandas. Telepítés: `pip install pandas`.")
        return
    if models is None:
        st.info("A tanító modul jelenleg nem elérhető. Ellenőrizd a függőségeket az alábbi szakaszban.")
        deps_warning_box()
        return

    upload_col, config_col = st.columns([2, 1])

    with upload_col:
        csv_file = st.file_uploader("CSV fájl feltöltése", type=["csv"]) 
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            st.write("Előnézet", df.head())
            target_column = st.selectbox("Cél oszlop (target)", options=list(df.columns))
            problem_type = st.selectbox("Feladat típusa", ["regresszió (számszerű)", "klasszifikáció (kategóriás)"])
            model_type = st.selectbox(
                "Modell",
                options=(
                    ["random_forest", "linear", "neural_network"] if problem_type.startswith("regresszió") else ["random_forest", "logistic", "neural_network"]
                ),
                index=0,
            )
            model_name = st.text_input("Modell neve", value=f"model_{int(datetime.now().timestamp())}")

            if st.button("Tanítás indítása", type="primary"):
                with st.spinner("Tanítás folyamatban..."):
                    try:
                        if problem_type.startswith("regresszió"):
                            results = models.train_numeric_model(df, target_column, model_name, model_type)
                        else:
                            results = models.train_categorical_model(df, target_column, model_name, model_type)
                        st.success("Sikeres tanítás!")
                        st.json(results)

                        # Opciós verzió létrehozás
                        if versions is not None:
                            with st.expander("Verzió készítése (opcionális)"):
                                version_name = st.text_input("Verzió megjegyzés / azonosító", value="initial")
                                if st.button("Verzió létrehozása"):
                                    perf_metrics: Dict[str, Any] = {k: v for k, v in results.items() if isinstance(v, (int, float, str))}
                                    model_path = Path("models") / (f"{model_name}.h5" if model_type == "neural_network" else f"{model_name}.pkl")
                                    version_id = versions.create_model_version(
                                        model_name=model_name,
                                        model_type=results.get("model_type", model_type),
                                        model_path=str(model_path),
                                        training_data=df,
                                        performance_metrics=perf_metrics,
                                        metadata={"note": version_name},
                                    )
                                    st.success(f"Verzió létrehozva: {version_id}")
                        else:
                            st.info("Model versioning nem elérhető (függőség hiányzik).")
                    except Exception as e:
                        st.error(f"Hiba: {e}")

    with config_col:
        st.caption("Tippek:")
        st.markdown("- Kis mintával kezdj, majd növeld.")
        st.markdown("- Gyors kipróbáláshoz válaszd a 'random_forest' modellt.")
        st.markdown("- A tanított modellek a `models/` mappába kerülnek.")

    st.divider()
    st.subheader("Modellek listája")
    try:
        if models is not None:
            model_list = models.list_models()
            if model_list:
                st.write(model_list)
            else:
                st.info("Még nincs elérhető modell.")
        else:
            st.info("Modellek listázása nem elérhető – telepítsd a függőségeket.")
    except Exception as e:
        st.warning(f"Modellek listázása sikertelen: {e}")


def ui_testing(models):
    st.subheader("Tesztelés / Generálás")

    if models is None:
        st.info("A teszt modul jelenleg nem elérhető. Ellenőrizd a függőségeket.")
        deps_warning_box()
        return

    try:
        available = models.list_models()
    except Exception:
        available = []

    if not available:
        st.info("Nincs modell. Előbb taníts egy modellt a Tanítás fülön.")
        return

    model_name = st.selectbox("Válassz modellt", options=available)

    # Automatikus betöltés, ha még nincs konfiguráció a memóriában
    if model_name not in getattr(models, 'configs', {}):
        try:
            models.load_model(model_name)
            st.caption("(Modell konfiguráció automatikusan betöltve)")
        except Exception as e:
            st.warning(f"Betöltési figyelmeztetés: {e}")

    with st.expander("Bemenet előkészítése"):
        num_rows = st.number_input("Minták száma", min_value=1, max_value=1000, value=5)
        input_mode = st.radio("Bemenet módja", ["Interaktív táblázat", "CSV feltöltés"], horizontal=True)

        input_df: Optional["pd.DataFrame"] = None  # type: ignore
        if input_mode == "CSV feltöltés":
            if pd is None:
                st.info("CSV beolvasáshoz szükséges a pandas (`pip install pandas`).")
            else:
                csv_in = st.file_uploader("Bemeneti CSV", type=["csv"], key="gen_csv")
                if csv_in is not None:
                    input_df = pd.read_csv(csv_in)
                    st.write("Előnézet", input_df.head())
        else:
            # Feature nevek lekérése, ha elérhető
            try:
                cfg = models.configs.get(model_name)
                cols = cfg.input_features if cfg else []
            except Exception:
                cols = []
            if not cols:
                st.info("Nem található konfiguráció. Adott számú oszloppal generálunk.")
                cols = [f"f{i+1}" for i in range(4)]
            # Interaktív szerkesztő pandas nélkül is működik (internálisan DataFrame-re konvertál)
            default_rows = [[0.0] * len(cols) for _ in range(num_rows)]
            default_df = pd.DataFrame(default_rows, columns=cols) if pd is not None else None
            input_df = st.data_editor(default_df, use_container_width=True, num_rows="dynamic")

    if st.button("Generálás / Predikció", type="primary"):
        with st.spinner("Generálás folyamatban..."):
            try:
                out_df = models.generate_data(
                    model_name,
                    num_samples=len(input_df) if input_df is not None else 10,
                    input_data=input_df,
                )
                st.success("Kész")
                st.write(out_df.head(50))

                if pd is not None:
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Eredmény letöltése CSV", data=csv_bytes, file_name=f"{model_name}_generated_{int(datetime.now().timestamp())}.csv")
            except Exception as e:
                st.error(f"Hiba: {e}")


def ui_adaptive(adaptive):
    st.subheader("Adaptív tanulás vezérlése")

    if adaptive is None:
        st.info("Az adaptív tanulási modul nem elérhető – telepítsd a numpy/pandas függőségeket.")
        deps_warning_box()
        return

    components = list(adaptive.learning_states.keys())
    component = st.selectbox("Komponens", options=components, index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Aktuális tanulási ráta", f"{adaptive.get_learning_rate(component):.6f}")
        if st.button("Ráta visszaállítása (base)"):
            adaptive.reset_learning_rate(component)
            st.success("Visszaállítva")

    with col2:
        st.caption("Teljesítménymutatók frissítése")
        acc = st.slider("Pontosság", 0.0, 1.0, 0.7, 0.01)
        loss = st.number_input("Veszteség", min_value=0.0, value=0.5, step=0.01)
        ttime = st.number_input("Tanítási idő (s)", min_value=0.0, value=1.0, step=0.1)
        conv = st.slider("Konvergencia ráta", 0.0, 1.0, 0.3, 0.01)
        stab = st.slider("Stabilitás", 0.0, 1.0, 0.8, 0.01)
        if st.button("Mérőszámok beküldése"):
            try:
                # Lazy import a dataclasshoz
                from src.ai_excel_learning.adaptive_learning import PerformanceMetrics  # type: ignore
                metrics = PerformanceMetrics(
                    accuracy=acc,
                    loss=loss,
                    training_time=ttime,
                    convergence_rate=conv,
                    stability_score=stab,
                    timestamp=datetime.now(),
                )
                info = adaptive.update_performance(component, metrics)
                st.success(info.get("reason", "Frissítve"))
                st.write(info)
            except Exception as e:
                st.error(f"Hiba: {e}")

    with st.expander("Ajánlások"):
        try:
            recs = adaptive.get_learning_recommendations(component)
            for r in recs:
                st.markdown(f"- {r}")
        except Exception as e:
            st.warning(f"Ajánlások nem elérhetők: {e}")


def ui_monitoring():
    st.subheader("Monitorozás / Metrikák")
    monitor = get_ai_monitor()

    with st.expander("Gyors metrika rögzítés"):
        # Komponensválasztás a fő komponensekből
        components = [
            "excel_analyzer",
            "chart_learner",
            "formula_learner",
            "ml_models",
            "learning_pipeline",
            "background_processor",
            "research_extensions",
        ]
        component = st.selectbox("Komponens", options=components, index=3)
        metric_type = st.selectbox("Metrika típusa", [m for m in MetricType])
        value = st.number_input("Érték", value=0.0, step=0.01)
        if st.button("Rögzítés"):
            try:
                monitor.record_metric(component=component, metric_type=metric_type, value=value, metadata={"source": "ui"})
                st.success("Rögzítve")
            except Exception as e:
                st.error(f"Hiba: {e}")

    with st.expander("Összefoglaló (utolsó 24 óra)", expanded=True):
        try:
            summary = monitor.get_performance_summary("ml_models", hours=24)
            st.json(summary)
        except Exception as e:
            st.warning(f"Összefoglaló nem elérhető: {e}")


def ui_labeling():
    st.subheader("Címkézés / Annotáció egyszerűen")
    st.caption("Tölts fel egy CSV-t, adj hozzá egy 'label' oszlopot, és mentsd el a 'labeled_datasets/' mappába.")

    if pd is None:
        st.info("A címkézéshez szükséges a pandas (`pip install pandas`).")
        return

    csv_file = st.file_uploader("CSV feltöltése", type=["csv"], key="label_csv")
    if csv_file is None:
        return

    df = pd.read_csv(csv_file)
    if "label" not in df.columns:
        df["label"] = ""

    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

    if st.button("Mentés labeled_datasets/ alá"):
        ensure_dir("labeled_datasets")
        out_path = Path("labeled_datasets") / f"labeled_{Path(csv_file.name).stem}_{int(datetime.now().timestamp())}.csv"
        edited.to_csv(out_path, index=False)
        st.success(f"Elmentve: {out_path}")


# --- App ---

def main():
    st.set_page_config(page_title="AI Lab", page_icon="🧪", layout="wide")
    st.title("🧪 AI Lab – Tanítás és Tesztelés")
    st.caption("Interaktív felület az AI komponensek gyors kipróbálásához és tanításához.")

    models = get_models_manager()
    versions = get_model_version_manager()
    adaptive = get_adaptive_system()

    tabs = st.tabs(["Tanítás", "Tesztelés", "Adaptív tanulás", "Monitorozás", "Címkézés", "Függőségek"])

    with tabs[0]:
        ui_training(models, versions)
    with tabs[1]:
        ui_testing(models)
    with tabs[2]:
        ui_adaptive(adaptive)
    with tabs[3]:
        ui_monitoring()
    with tabs[4]:
        ui_labeling()
    with tabs[5]:
        deps_warning_box()


if __name__ == "__main__":
    main()