"""Streamlit dashboard for the crop disease pipeline.

Five tabs, one per graded capability:

1. **Predict**      -- upload one leaf image, get a classification
2. **Monitoring**   -- API uptime, latency percentiles, throughput, drift state
3. **Data insights** -- the EDA features and what they actually tell us
4. **Upload data**  -- stage bulk labelled images for retraining
5. **Retrain**      -- trigger retraining, watch it run, compare and roll back

The UI holds no model. Everything goes through the API, so what is on screen is
the deployed service's real state rather than a local recomputation -- which is
the point of the monitoring tab.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui import theme

def _resolve_api_url() -> str:
    """Find the API base URL: Streamlit secret, then env var, then localhost.

    Two quirks are handled here:

    * `st.secrets` *raises* rather than returning empty when no secrets.toml
      exists -- the normal case locally and in Docker, where the URL arrives as
      an environment variable -- so the lookup must be guarded.
    * Render's `fromService` injects a bare hostname with no scheme, and it
      does not interpolate strings into env vars. A scheme is added when
      missing, defaulting to https for anything that is not local.
    """
    raw = ""
    try:
        raw = str(st.secrets.get("API_URL") or "")
    except Exception:
        raw = ""
    if not raw:
        raw = os.getenv("API_URL", "http://localhost:8000")

    raw = raw.strip().rstrip("/")
    if not raw.startswith(("http://", "https://")):
        local = raw.startswith(("localhost", "127.0.0.1", "nginx", "api"))
        raw = f"{'http' if local else 'https'}://{raw}"
    return raw


API_URL = _resolve_api_url()

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
EDA_FEATURES_CSV = Path(__file__).resolve().parent.parent / "reports" / "eda_features.csv"
REQUEST_TIMEOUT = 30

st.set_page_config(
    page_title="Crop Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# API helpers
# ==========================================================================

def api_get(path: str, **kwargs):
    """GET with a uniform failure shape.

    Returns (payload, error). The UI must stay usable when the API is down --
    an unreachable backend is exactly the condition the monitoring tab exists
    to display, so it cannot be an exception that blanks the page.
    """
    try:
        response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}: {response.text[:300]}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"cannot reach API at {API_URL} ({exc.__class__.__name__})"


def api_post(path: str, **kwargs):
    try:
        response = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}: {response.text[:300]}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"cannot reach API at {API_URL} ({exc.__class__.__name__})"


@st.cache_data(ttl=300)
def fetch_classes() -> list[str]:
    payload, _ = api_get("/classes")
    return [c["class_name"] for c in payload["classes"]] if payload else []


def stat_tile(column, label: str, value: str, caption: str = "") -> None:
    """A single number is a stat tile, not a one-bar chart."""
    with column:
        st.metric(label, value)
        if caption:
            st.caption(caption)


# ==========================================================================
# Sidebar
# ==========================================================================

with st.sidebar:
    st.markdown("### 🌿 Crop Disease Classifier")
    st.caption("38 classes · 14 crop species")

    health, health_error = api_get("/health")

    if health_error:
        st.error("API unreachable")
        st.caption(health_error)
    elif health["model_ready"]:
        st.success(f"API healthy · model {health['model_version']}")
    else:
        st.warning("API up, model not loaded")
        st.caption("Run the notebook's export step to produce artifacts.")

    if health:
        st.caption(f"Uptime {health['uptime_human']} · {health['total_predictions']:,} predictions")
        st.caption(f"Replica `{health['hostname']}`")

    st.divider()
    st.caption(f"API: `{API_URL}`")
    st.link_button("OpenAPI docs", f"{API_URL}/docs", use_container_width=True)


tab_predict, tab_monitor, tab_insights, tab_upload, tab_retrain = st.tabs([
    "Predict", "Monitoring", "Data insights", "Upload data", "Retrain",
])


# ==========================================================================
# 1. Predict
# ==========================================================================

with tab_predict:
    st.subheader("Classify a leaf image")
    st.caption(
        "Upload a photograph of a crop leaf. The image runs through a frozen "
        "MobileNetV2 backbone and a classifier head to produce a disease diagnosis."
    )

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Leaf image", type=["jpg", "jpeg", "png"],
            help="A single leaf, ideally filling most of the frame.",
        )
        top_k = st.slider("Alternatives to show", 3, 10, 5)
        if uploaded is not None:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

    with right:
        if uploaded is None:
            st.info("Upload an image to see a prediction.")
        else:
            with st.spinner("Classifying..."):
                result, error = api_post(
                    f"/predict?top_k={top_k}",
                    files={"file": (uploaded.name, uploaded.getvalue(),
                                    uploaded.type or "image/png")},
                )

            if error:
                st.error(error)
            else:
                colour, status_label = theme.status_colour(
                    result["healthy"], result["confidence"])

                st.markdown(
                    f"<div style='padding:16px 18px;border-radius:10px;"
                    f"border:1px solid rgba(128,128,128,0.22);'>"
                    f"<div style='font-size:13px;color:{colour};font-weight:600;"
                    f"letter-spacing:0.02em;'>"
                    f"{'●' if result['healthy'] else '▲'} {status_label}</div>"
                    f"<div style='font-size:26px;font-weight:650;margin-top:4px;'>"
                    f"{result['crop']}</div>"
                    f"<div style='font-size:17px;opacity:0.75;'>{result['condition']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                cols = st.columns(3)
                stat_tile(cols[0], "Confidence", f"{result['confidence']:.1%}")
                stat_tile(cols[1], "Latency", f"{result['latency_ms']:.0f} ms")
                stat_tile(cols[2], "Model", result["model_version"] or "—")

                st.markdown("**Ranked alternatives**")
                st.plotly_chart(
                    theme.confidence_chart(result["top_k"]),
                    use_container_width=True, config={"displayModeBar": False},
                )

                # The table view is the relief for chart colours that sit below
                # the contrast floor, and it carries the exact numbers.
                with st.expander("View as table"):
                    st.dataframe(
                        pd.DataFrame([
                            {"Rank": p["rank"], "Crop": p["crop"],
                             "Condition": p["condition"],
                             "Probability": f"{p['probability']:.4f}"}
                            for p in result["top_k"]
                        ]),
                        hide_index=True, use_container_width=True,
                    )


# ==========================================================================
# 2. Monitoring
# ==========================================================================

with tab_monitor:
    header, refresh = st.columns([4, 1])
    header.subheader("Service health and performance")
    if refresh.button("Refresh", use_container_width=True):
        st.rerun()

    metrics, metrics_error = api_get("/metrics")

    if metrics_error:
        st.error(metrics_error)
    else:
        st.caption(
            f"Serving replica `{metrics['hostname']}` · started {metrics['started_at']}"
        )

        cols = st.columns(4)
        stat_tile(cols[0], "Uptime", metrics["uptime_human"])
        stat_tile(cols[1], "Predictions", f"{metrics['total_predictions']:,}")
        stat_tile(cols[2], "Errors", f"{metrics['total_errors']:,}",
                  f"{metrics['error_rate']:.2%} of requests")
        stat_tile(cols[3], "Throughput", f"{metrics['requests_per_second_1m']:.2f} req/s",
                  "last 60 s")

        st.markdown("#### Latency")
        st.caption(
            "Percentiles matter more than the mean here: the p99 is what a user "
            "on a slow request actually experiences."
        )
        latency = metrics["latency_ms"]
        cols = st.columns(4)
        stat_tile(cols[0], "p50", f"{latency['p50']:.0f} ms")
        stat_tile(cols[1], "p95", f"{latency['p95']:.0f} ms")
        stat_tile(cols[2], "p99", f"{latency['p99']:.0f} ms")
        stat_tile(cols[3], "max", f"{latency['max']:.0f} ms")

        st.divider()
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("#### Drift signal")
            confidence = metrics["confidence"]
            if confidence["rolling_mean"] is None:
                st.info("No predictions yet in the drift window.")
            else:
                delta = confidence["rolling_mean"] - confidence["threshold"]
                st.metric(
                    "Rolling mean confidence",
                    f"{confidence['rolling_mean']:.3f}",
                    f"{delta:+.3f} vs threshold",
                    delta_color="normal" if delta >= 0 else "inverse",
                )
                st.caption(
                    f"Averaged over the last {confidence['window_size']} predictions. "
                    f"Falling below {confidence['threshold']:.2f} triggers retraining."
                )

            trigger, _ = api_get("/retrain/trigger")
            if trigger:
                if trigger["should_retrain"]:
                    st.warning("Retraining conditions met")
                    for reason in trigger["reasons"]:
                        st.caption(f"• {reason}")
                else:
                    st.success("No retraining needed")

        with right:
            st.markdown("#### Most-predicted classes")
            top_classes = metrics["top_predicted_classes"]
            if not top_classes:
                st.info("No predictions recorded yet.")
            else:
                labels = [c["class_name"].replace("___", " — ").replace("_", " ")
                          for c in top_classes]
                st.plotly_chart(
                    theme.ranked_bar(labels, [c["count"] for c in top_classes],
                                     axis_title="predictions"),
                    use_container_width=True, config={"displayModeBar": False},
                )

        if metrics.get("model_metrics"):
            st.divider()
            st.markdown("#### Active model quality")
            st.caption(
                "Measured on the held-out test set at training time. Macro-F1 is "
                "the headline: it weights rare disease classes equally with the "
                "abundant tomato ones."
            )
            model_metrics = {k: v for k, v in metrics["model_metrics"].items()
                             if v is not None}
            cols = st.columns(min(len(model_metrics), 4) or 1)
            for i, (name, value) in enumerate(model_metrics.items()):
                stat_tile(cols[i % len(cols)],
                          name.replace("_", " ").title(), f"{value:.4f}")


# ==========================================================================
# 3. Data insights
# ==========================================================================

with tab_insights:
    st.subheader("What the data actually tells us")
    st.caption(
        "Three quantitative findings that shaped the model, plus the one that "
        "shaped the whole pipeline."
    )

    if EDA_FEATURES_CSV.exists():
        features = pd.read_csv(EDA_FEATURES_CSV)
    else:
        features = None
        st.info(
            "Interactive charts appear once the notebook has been run — it writes "
            f"`{EDA_FEATURES_CSV.relative_to(EDA_FEATURES_CSV.parents[1])}`. "
            "The interpretations below are still the findings that drove the design."
        )

    # -- Finding 1 --------------------------------------------------------
    st.markdown("#### 1 · Class imbalance decides which metric we trust")
    st.markdown(
        "Class sizes span roughly **150 to 5,500 images**, a ~36× ratio, and tomato "
        "alone accounts for 10 of the 38 classes. A model that ignored every rare "
        "class entirely would still post a respectable accuracy. That makes accuracy "
        "the wrong headline number — **macro-F1** weights each class equally, so it "
        "notices exactly the failure that matters here: missing a rare disease on a "
        "crop the bank has lent against."
    )
    if features is not None and "class" in features:
        counts = (features.groupby(["crop", "healthy"]).size()
                  .reset_index(name="count"))
        pivot = counts.pivot(index="crop", columns="healthy", values="count").fillna(0)
        pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
        st.plotly_chart(
            theme.ranked_bar(list(pivot.index), list(pivot.sum(axis=1)),
                             axis_title="images sampled"),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.divider()

    # -- Finding 2 --------------------------------------------------------
    st.markdown("#### 2 · Colour carries the signal — but only within a crop")
    st.markdown(
        "Chlorosis and necrosis — yellowing and browning — are the visible signature "
        "of most foliar disease, so the **excess-green index** (2G − R − B, standard in "
        "precision agriculture) ought to separate healthy from diseased leaves. "
        "Pooled across all 14 crops it barely does: **|Cohen's d| = 0.25**, which is a "
        "small effect. Hold the crop constant and the same feature becomes the "
        "strongest measured, at **|d| = 1.15** — a 4.5× jump."
    )
    st.markdown(
        "The direction is crop-specific too: diseased corn reads *greener* than healthy "
        "corn, while diseased apple reads browner. **No single global colour threshold "
        "can work.** That is the empirical case for a learned, crop-conditional "
        "representation rather than hand-crafted rules — and it is why the pipeline "
        "uses a pretrained backbone instead of the vegetation indices directly."
    )
    if features is not None and {"excess_green", "redness_index"} <= set(features.columns):
        sample = features.sample(min(len(features), 1500), random_state=0).copy()
        sample["Leaf"] = sample["healthy"].map({True: "Healthy", False: "Diseased"})
        st.plotly_chart(
            theme.grouped_scatter(sample, "excess_green", "redness_index", "Leaf",
                                  x_title="excess green (2G − R − B)",
                                  y_title="redness index"),
            use_container_width=True, config={"displayModeBar": False},
        )
        with st.expander("View group means as a table"):
            st.dataframe(
                sample.groupby("Leaf")[["excess_green", "redness_index",
                                        "green_ratio", "mean_saturation"]]
                .mean().round(3).reset_index(),
                hide_index=True, use_container_width=True,
            )

    st.divider()

    # -- Finding 3 --------------------------------------------------------
    st.markdown("#### 3 · Two texture features that turned out to be one — and weak")
    st.markdown(
        "The hypothesis was that necrotic spots and rust pustules introduce "
        "high-frequency transitions a smooth leaf lacks, giving a texture signal "
        "independent of colour. **The data refuted both halves of that.**"
    )
    st.markdown(
        "**They are not independent of each other.** Edge density and Laplacian "
        "variance correlate at **r = 0.90** — one signal, not two. Keeping both would "
        "have added a feature and no information.\n\n"
        "**And pooled, the signal is close to absent.** Edge density separates healthy "
        "from diseased at **|d| = 0.10**, with the sign *opposite* to the prediction. "
        "Only after conditioning on crop does it reach |d| = 0.77 — the same lesson as "
        "finding 2, which is the real result here: **crop identity dominates the "
        "feature space, and disease is the finer-grained axis underneath it.**"
    )
    if features is not None and {"edge_density", "laplacian_var"} <= set(features.columns):
        sample = features.sample(min(len(features), 1500), random_state=0).copy()
        sample["Leaf"] = sample["healthy"].map({True: "Healthy", False: "Diseased"})
        st.plotly_chart(
            theme.grouped_scatter(sample, "edge_density", "laplacian_var", "Leaf",
                                  x_title="edge density (fraction of edge pixels)",
                                  y_title="Laplacian variance"),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.caption(
            "The near-linear cloud is the redundancy: the two axes measure the same "
            "thing. Healthy and diseased overlap almost completely."
        )
        st.warning(
            "**A negative result kept rather than dropped.** Reporting only the "
            "features that worked would misrepresent how well hand-crafted features "
            "do on this problem — and would hide the finding that motivates the "
            "whole architecture."
        )

    st.divider()

    # -- Finding 4 --------------------------------------------------------
    st.markdown("#### 4 · The background is uniform — and that is a warning")
    st.markdown(
        "Every PlantVillage image is a detached leaf photographed on a **uniform "
        "laboratory background**. Corner-pixel variance confirms it quantitatively. "
        "This is the most consequential finding in the whole project, and it is a "
        "caveat rather than a feature: a model scoring 99% here is partly reading "
        "*acquisition conditions*, not just disease, and it will degrade on a loan "
        "officer's photo taken in a field with soil, hands and mixed daylight in frame."
    )
    st.markdown(
        "There is direct evidence for that in the numbers. Pooled across crops, the "
        "single best separator of healthy from diseased is **mean brightness** "
        "(|d| = 0.76) — ahead of every vegetation index. Brightness carries no "
        "botanical meaning; a leaf is not diseased because the photograph is darker. "
        "A lighting or capture-session artefact out-ranking the biology is precisely "
        "the signature of a dataset shortcut, and it is what a field photo will not "
        "reproduce."
    )
    st.info(
        "**This is why the pipeline retrains.** Expected degradation on real field "
        "photos is the reason for the confidence-drift trigger and the upload-and-"
        "retrain loop — the deployed model is built to absorb a domain shift it is "
        "known to face, rather than assuming the test score transfers."
    )
    if features is not None and "background_fraction" in features.columns:
        with st.expander("Background uniformity measurements"):
            st.dataframe(
                features[["corner_std", "background_fraction", "corner_brightness"]]
                .describe().round(3).reset_index(),
                hide_index=True, use_container_width=True,
            )

    # -- Static figures ---------------------------------------------------
    figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if figures:
        st.divider()
        st.markdown("#### Figures from the notebook")
        columns = st.columns(2)
        for i, figure in enumerate(figures):
            with columns[i % 2]:
                st.image(str(figure), use_container_width=True,
                         caption=figure.stem.replace("_", " "))


# ==========================================================================
# 4. Upload data
# ==========================================================================

with tab_upload:
    st.subheader("Stage new labelled images")
    st.caption(
        "Uploaded images are held until a retrain consumes them. A label is "
        "required — inferring it from filenames would let bad data into the "
        "replay buffer, where it would corrupt every future retrain."
    )

    classes = fetch_classes()
    if not classes:
        st.warning("Class list unavailable — the API has no model artifacts yet.")
    else:
        left, right = st.columns([1.3, 1], gap="large")

        with left:
            crops = sorted({c.split("___")[0].replace("_", " ").replace(",", "")
                            for c in classes})
            crop = st.selectbox("Crop", crops)
            matching = [c for c in classes
                        if c.split("___")[0].replace("_", " ").replace(",", "") == crop]
            label = st.selectbox(
                "Condition", matching,
                format_func=lambda c: c.split("___")[-1].replace("_", " "),
            )

            files = st.file_uploader(
                "Images for this class", type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="All files uploaded together are labelled with the class above.",
            )

            if files and st.button(f"Upload {len(files)} image(s)",
                                   type="primary", use_container_width=True):
                with st.spinner(f"Uploading {len(files)} images..."):
                    payload = [("files", (f.name, f.getvalue(), f.type or "image/png"))
                               for f in files]
                    result, error = api_post("/upload", files=payload,
                                             data={"class_name": label})

                if error:
                    st.error(error)
                else:
                    st.success(f"Staged {result['accepted']} image(s) as `{label}`")
                    if result["rejected"]:
                        st.warning(f"{result['rejected']} rejected")
                        for message in result["errors"][:10]:
                            st.caption(f"• {message}")
                    if result["retrain_trigger"]["should_retrain"]:
                        st.info("Enough data staged — retraining is now available.")
                    st.cache_data.clear()

        with right:
            st.markdown("#### Currently staged")
            summary, error = api_get("/uploads/summary")
            if error:
                st.error(error)
            elif summary["total_images"] == 0:
                st.info("Nothing staged yet.")
            else:
                cols = st.columns(2)
                stat_tile(cols[0], "Images", f"{summary['total_images']:,}")
                stat_tile(cols[1], "Classes", str(summary["n_classes"]))

                st.dataframe(
                    pd.DataFrame([
                        {"Class": c["class_name"].replace("___", " — ").replace("_", " "),
                         "Images": c["count"]}
                        for c in summary["per_class"]
                    ]),
                    hide_index=True, use_container_width=True,
                )

                trigger = summary["retrain_trigger"]["triggers"]["volume"]
                st.progress(
                    min(trigger["staged_images"] / max(trigger["threshold"], 1), 1.0),
                    text=f"{trigger['staged_images']} / {trigger['threshold']} "
                         "images to auto-trigger",
                )


# ==========================================================================
# 5. Retrain
# ==========================================================================

with tab_retrain:
    st.subheader("Retrain the classifier")
    st.caption(
        "Retraining embeds the staged images through the frozen backbone and "
        "refits only the classifier head — seconds, not hours, and inside the "
        "deployed container. The backbone is never touched."
    )

    trigger, trigger_error = api_get("/retrain/trigger")

    if trigger_error:
        st.error(trigger_error)
    else:
        left, right = st.columns([1, 1.4], gap="large")

        with left:
            volume = trigger["triggers"]["volume"]
            drift = trigger["triggers"]["drift"]

            st.markdown("#### Trigger status")
            st.caption(
                f"**Volume** — {volume['staged_images']} staged, "
                f"threshold {volume['threshold']}"
            )
            st.caption(
                "**Drift** — rolling confidence "
                + (f"{drift['rolling_mean_confidence']:.3f}"
                   if drift["rolling_mean_confidence"] is not None else "n/a")
                + f", threshold {drift['threshold']:.2f}"
                + ("" if drift["window_filled"] else " (window not yet full)")
            )

            if trigger["should_retrain"]:
                st.warning("Retraining conditions met")
            else:
                st.success("Conditions not met")

            force = st.checkbox(
                "Force retrain", value=not trigger["should_retrain"],
                help="Retrain even when the automatic conditions are unmet.",
            )
            keep = st.checkbox(
                "Keep uploads staged", value=False,
                help="Normally consumed uploads move into the training set.",
            )

            disabled = not (trigger["should_retrain"] or force)
            if st.button("Trigger retraining", type="primary",
                         use_container_width=True, disabled=disabled):
                job, error = api_post(
                    f"/retrain?force={str(force).lower()}"
                    f"&keep_uploads={str(keep).lower()}")
                if error:
                    st.error(error)
                else:
                    st.session_state["retrain_job"] = job["job_id"]
                    st.rerun()

        with right:
            job_id = st.session_state.get("retrain_job")
            if job_id:
                st.markdown("#### Job progress")
                placeholder = st.empty()

                # Poll until the job settles. Retraining is seconds-scale, so a
                # short bounded poll is simpler than a websocket and cannot hang
                # the page.
                for _ in range(120):
                    job, error = api_get(f"/retrain/status/{job_id}")
                    if error or job is None:
                        placeholder.error(error or "job vanished")
                        break

                    with placeholder.container():
                        st.progress(job["progress"], text=job["message"])
                    if job["status"] != "running":
                        break
                    time.sleep(1)

                if job and job["status"] == "completed":
                    result = job["result"]
                    if result["promoted"]:
                        st.success(f"Promoted **{result['version']}** — {result['reason']}")
                    else:
                        st.warning(
                            f"**{result['version']}** registered but NOT promoted — "
                            f"{result['reason']}"
                        )
                        st.caption(
                            "The registry refuses to activate a head that regresses "
                            "macro-F1 beyond tolerance. The previous model is still live."
                        )

                    cols = st.columns(3)
                    stat_tile(cols[0], "New images", f"{result['n_new_images']:,}")
                    stat_tile(cols[1], "Replay buffer", f"{result['n_replay_images']:,}")
                    stat_tile(cols[2], "Duration", f"{result['duration_seconds']:.1f} s")

                    before, after = result["metrics_before"], result["metrics_after"]
                    keys = [k for k in ("accuracy", "f1_macro", "top5_accuracy")
                            if before.get(k) is not None and after.get(k) is not None]
                    if keys:
                        st.markdown("**Before vs after**")
                        st.plotly_chart(
                            theme.comparison_chart(
                                [k.replace("_", " ") for k in keys],
                                [before[k] for k in keys],
                                [after[k] for k in keys]),
                            use_container_width=True, config={"displayModeBar": False},
                        )
                elif job and job["status"] == "failed":
                    st.error(f"Retraining failed: {job['error']}")
            else:
                st.info("Trigger a retrain to see progress here.")

    st.divider()
    st.markdown("#### Model versions")
    st.caption(
        "Every retrain produces a new version rather than overwriting. A head is "
        "a few hundred KB, so keeping the history is essentially free — and it "
        "makes rollback instant."
    )

    models, models_error = api_get("/models")
    if models_error:
        st.error(models_error)
    elif not models["versions"]:
        st.info("No model versions registered yet.")
    else:
        st.dataframe(
            pd.DataFrame([
                {
                    "Active": "✓" if v["active"] else "",
                    "Version": v["version"],
                    "Created": v["created_at"],
                    "Source": v["source"],
                    "Accuracy": f"{v['accuracy']:.4f}" if v["accuracy"] is not None else "—",
                    "Macro-F1": f"{v['f1_macro']:.4f}" if v["f1_macro"] is not None else "—",
                    "Train size": f"{v['n_train_samples']:,}",
                    "New": f"{v['n_new_samples']:,}",
                }
                for v in models["versions"]
            ]),
            hide_index=True, use_container_width=True,
        )

        inactive = [v["version"] for v in models["versions"] if not v["active"]]
        if inactive:
            left, right = st.columns([2, 1])
            target = left.selectbox("Roll back to", inactive, label_visibility="collapsed")
            if right.button("Activate", use_container_width=True):
                result, error = api_post(f"/models/activate/{target}")
                if error:
                    st.error(error)
                else:
                    st.success(f"Activated {target}")
                    st.rerun()
