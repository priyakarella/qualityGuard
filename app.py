"""
QualityGuard Streamlit UI - Product Quality Assessment System.

Main entry point for the quality assessment application.
Orchestrates product quality analysis with ML models and multi-agent reasoning.
"""
# to avoid ssl certificate error
import ssl
import os

# Corporate laptop SSL fix
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.gemini_client import initialize_gemini_llm
from flows.quality_assessment_flow import QualityAssessmentFlow
from config.constants import (
    ASSESSMENT_TABS, PRODUCTS_FILE, DATA_ASSESSMENTS_DIR, DATA_FEEDBACK_DIR,
    ML_FEATURES_ORDERED, CATEGORICAL_FEATURES, VALID_COUNTRIES,
    STATUS_APPROVED, STATUS_REJECTED, STATUS_PENDING_REVIEW
)


# TODO: Implement all UI functions and main() entry point
# Purpose: Create Streamlit application with 4 main tabs: New Assessment, Assessment History, Feedback Analysis, Model Evaluation
# Functions needed: initialize_session(), load_sample_products() with feature mapping, display_assessment_results(),
# save_assessment(), load_all_assessments(), display_assessment_history(), display_feedback_analysis(),
# display_ai_feedback_results(), generate_insights(), evaluate_defect_classifier(), evaluate_return_rate_predictor(),
# display_model_evaluation_tab(), main() with sidebar configuration and Streamlit page setup
# Returns: Interactive UI with assessment workflow, human approval gate, feedback collection, history tracking, model evaluation

import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.constants import (
    ASSESSMENT_TABS,
    DATA_ASSESSMENTS_DIR,
    DATA_FEEDBACK_DIR,
    ML_FEATURES_ORDERED,
    PRODUCTS_FILE,
    STATUS_APPROVED,
    STATUS_PENDING_REVIEW,
    STATUS_REJECTED,
    VALID_COUNTRIES,
)
from flows.quality_assessment_flow import QualityAssessmentFlow
from utils.gemini_client import initialize_gemini_llm
# ── Arize Phoenix tracing (auto-instruments CrewAI agents) ────────────────────
def _init_tracing():
    """Connect to Phoenix if it's running. Silently skip if not."""
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        import requests

        # Only connect if Phoenix is actually running
        try:
            requests.get("http://localhost:6006", timeout=2)
        except Exception:
            return   # Phoenix not running — skip silently

        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
            )
        )
        otel_trace.set_tracer_provider(tracer_provider)
        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
        print("✅ Phoenix tracing connected — view at http://localhost:6006")
    except Exception:
        pass   # tracing is optional, never crash the app

_init_tracing()


# ─────────────────────────────────────────────────────────────────────────────
# Startup check
# ─────────────────────────────────────────────────────────────────────────────

def check_models_exist():
    """Stop app with clear message if ML models are missing."""
    models_dir = Path(__file__).parent / "ml" / "models"
    required = [
        "defect_classifier.pkl", "defect_scaler.pkl", "defect_encoder.pkl",
        "return_predictor.pkl",  "return_scaler.pkl",  "return_encoder.pkl",
    ]
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        st.error(f"⚠ ML model files not found: {missing}")
        st.info("Train the models first by running:\n```\npython ml/train_pipeline.py\n```")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def initialize_session():
    defaults = {
        "assessments":            [],
        "current_assessment":     None,
        "human_decision":         None,
        "ai_feedback_cache":      None,
        "feedback_stale":         False,
        "eval_defect_metrics":    None,
        "eval_return_metrics":    None,
        "assessment_running":     False,
        "last_selected_product":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sample_products():
    with open(PRODUCTS_FILE) as f:
        products = json.load(f)
    mapped = []
    for p in products:
        mapped.append({
            "name":                          p["name"],
            "category":                      p.get("category", ""),
            "product_price_usd":             p["price"],
            "warranty_period_months":        p["warranty"],
            "customer_review_count":         p["reviews"],
            "customer_average_rating":       p["rating"],
            "material_quality_score":        p["material_quality"],
            "supplier_reliability_rating":   p["supplier_rating"],
            "product_age_days":              p["product_age"],
            "product_manufacturing_country": p["country"].upper(),
            "market_demand_index":           p["market_demand"],
            "inventory":                     p.get("inventory", 1000),
        })
    return mapped


def save_assessment(results: dict, status: str, feedback: str):
    Path(DATA_ASSESSMENTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_FEEDBACK_DIR).mkdir(parents=True, exist_ok=True)
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    aid = results.get("assessment_id", uid)

    assessment_file = Path(DATA_ASSESSMENTS_DIR) / f"{status}_{ts}_{aid}.json"
    with open(assessment_file, "w") as f:
        json.dump(
            {**results, "human_status": status, "human_feedback": feedback, "assessment_id": aid},
            f, indent=2, default=str,
        )

    fb_file = Path(DATA_FEEDBACK_DIR) / f"fb_{ts}_{aid}.json"
    with open(fb_file, "w") as f:
        json.dump({
            "assessment_id":      aid,
            "product_name":       results.get("product_name"),
            "status":             status.upper(),
            "decision":           "APPROVED" if status == STATUS_APPROVED else "REJECTED",
            "feedback":           feedback,
            "timestamp":          datetime.utcnow().isoformat(),
            "defect_probability": results.get("defect_analysis", {}).get("defect_probability"),
            "return_rate":        results.get("quality_assessment", {}).get("return_rate"),
            "final_quality_grade":results.get("recommendation", {}).get("final_quality_grade"),
        }, f, indent=2, default=str)

    st.session_state["ai_feedback_cache"] = None
    st.session_state["feedback_stale"]    = True


def load_all_assessments():
    d = Path(DATA_ASSESSMENTS_DIR)
    if not d.exists():
        return []
    records = []
    for f in d.glob("*.json"):
        try:
            with open(f) as fp:
                records.append(json.load(fp))
        except Exception:
            continue
    return sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Assessment results display
# ─────────────────────────────────────────────────────────────────────────────

GRADE_ICON = {"EXCELLENT": "🟢", "GOOD": "🟡", "ACCEPTABLE": "🟠", "POOR": "🔴"}


def display_assessment_results(results: dict):
    if not results:
        return

    rec   = results.get("recommendation", {})
    da    = results.get("defect_analysis", {})
    qa    = results.get("quality_assessment", {})
    grade = rec.get("final_quality_grade", "N/A")
    icon  = GRADE_ICON.get(str(grade).upper(), "⚪")

    st.markdown(
        f"## {icon} Quality Grade: **{grade}**  |  "
        f"Status: `{rec.get('approval_status', 'PENDING_REVIEW')}`"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Defect Probability", f"{da.get('defect_probability', 'N/A')}%")
    c2.metric("Return Rate",        f"{qa.get('return_rate', 'N/A')}%")
    c3.metric("Quality Score",      f"{qa.get('quality_score', 'N/A')}/100")
    c4.metric("Risk Level",         da.get("risk_level", "N/A"))

    st.download_button(
        label="⬇ Download Assessment JSON",
        data=json.dumps(results, indent=2, default=str),
        file_name=f"assessment_{results.get('product_name','unknown')}_{results.get('timestamp','')[:10]}.json",
        mime="application/json",
    )

    tabs = st.tabs(ASSESSMENT_TABS)

    with tabs[0]:
        st.subheader("🔧 Defect Analysis")
        st.info(da.get("analysis_summary", "No summary."))
        col1, col2 = st.columns(2)
        col1.metric("Defect Probability", f"{da.get('defect_probability', 'N/A')}%")
        col2.metric("Risk Level", da.get("risk_level", "N/A"))

    with tabs[1]:
        st.subheader("📊 Quality Assessment")
        st.info(qa.get("assessment_summary", "No summary."))
        col1, col2 = st.columns(2)
        col1.metric("Return Rate",   f"{qa.get('return_rate', 'N/A')}%")
        col2.metric("Quality Score", f"{qa.get('quality_score', 'N/A')}/100")

    with tabs[2]:
        sc = results.get("supply_chain_analysis", {})
        st.subheader("🏭 Supply Chain Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Country",     sc.get("supplier_country", "N/A"))
        col2.metric("Reliability", f"{sc.get('supplier_reliability', 'N/A')}/5")
        col3.metric("Lead Time",   f"{sc.get('lead_time_days', 'N/A')} days")
        certs = sc.get("certifications", [])
        if certs:
            st.write("**Certifications:**", ", ".join(certs) if isinstance(certs, list) else str(certs))
        st.info(sc.get("supply_chain_assessment", "N/A"))

    with tabs[3]:
        ci = results.get("cost_impact_analysis", {})
        st.subheader("💰 Cost Impact")
        col1, col2, col3 = st.columns(3)
        col1.metric("Defective Units", ci.get("estimated_defective_units", "N/A"))
        col2.metric("Returned Units",  ci.get("estimated_returned_units",  "N/A"))
        fim = ci.get("total_financial_impact", 0)
        col3.metric("Total Impact", f"${float(fim):,.0f}" if fim else "N/A")
        st.info(ci.get("cost_breakdown", "N/A"))

    with tabs[4]:
        mt = results.get("market_trend_analysis", {})
        st.subheader("📈 Market Trends")
        col1, col2 = st.columns(2)
        col1.metric("Market Demand",        mt.get("market_demand", "N/A"))
        col2.metric("Competitive Position", mt.get("competitive_position", "N/A"))
        st.info(mt.get("trend_analysis", "N/A"))

    with tabs[5]:
        st.subheader("✅ Final Recommendation")
        st.write(rec.get("overall_assessment", "N/A"))
        recs = rec.get("recommendations", [])
        if recs:
            for i, r in enumerate(recs, 1):
                st.markdown(f"**{i}.** {r}")

    with tabs[6]:
        st.subheader("🐛 Debug / Raw Agent Outputs")
        for key in ["_raw_defect", "_raw_quality", "_raw_supply", "_raw_cost", "_raw_market", "_raw_rec"]:
            raw = results.get(key, "")
            label = key.replace("_raw_", "").title()
            with st.expander(f"Agent: {label}"):
                st.text(str(raw)[:3000])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Assessment History
# ─────────────────────────────────────────────────────────────────────────────

def display_assessment_history():
    st.header("📋 Assessment History")
    records = load_all_assessments()
    if not records:
        st.info("No assessments yet. Run a New Assessment first.")
        return

    col1, col2 = st.columns([3, 1])
    search   = col1.text_input("🔍 Search product name", "")
    f_status = col2.selectbox("Filter status", ["All", STATUS_APPROVED, STATUS_REJECTED, STATUS_PENDING_REVIEW])

    if search:
        records = [r for r in records if search.lower() in str(r.get("product_name", "")).lower()]
    if f_status != "All":
        records = [r for r in records if str(r.get("human_status", "")).upper() == f_status]

    st.caption(f"Showing {len(records)} record(s)")

    # Grade distribution chart
    all_grades = [r.get("recommendation", {}).get("final_quality_grade", "Unknown") for r in records]
    if all_grades:
        grade_counts = pd.Series(all_grades).value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        grade_order = ["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR", "Unknown"]
        grade_counts["Grade"] = pd.Categorical(grade_counts["Grade"], categories=grade_order, ordered=True)
        grade_counts = grade_counts.sort_values("Grade")
        st.bar_chart(grade_counts.set_index("Grade"), color="#4CAF50")

    # Comparison view
    all_names = [r.get("product_name", "Unknown") for r in records]
    if len(all_names) >= 2:
        with st.expander("🔀 Compare Two Assessments"):
            selected = st.multiselect(
                "Select exactly 2 products to compare:",
                options=list(dict.fromkeys(all_names)),
                max_selections=2,
            )
            if len(selected) == 2:
                r1 = next((r for r in records if r.get("product_name") == selected[0]), {})
                r2 = next((r for r in records if r.get("product_name") == selected[1]), {})
                col1, col2 = st.columns(2)
                for col, rec, name in [(col1, r1, selected[0]), (col2, r2, selected[1])]:
                    da   = rec.get("defect_analysis", {})
                    qa   = rec.get("quality_assessment", {})
                    reco = rec.get("recommendation", {})
                    grade = reco.get("final_quality_grade", "N/A")
                    with col:
                        st.subheader(f"{GRADE_ICON.get(str(grade).upper(),'⚪')} {name}")
                        st.metric("Defect %",      da.get("defect_probability", "N/A"))
                        st.metric("Return %",      qa.get("return_rate", "N/A"))
                        st.metric("Quality Score", qa.get("quality_score", "N/A"))
                        st.metric("Grade",         grade)
                        st.metric("Risk",          da.get("risk_level", "N/A"))

    st.divider()

    for rec in records:
        da    = rec.get("defect_analysis", {})
        qa    = rec.get("quality_assessment", {})
        reco  = rec.get("recommendation", {})
        grade = reco.get("final_quality_grade", "N/A")
        icon  = GRADE_ICON.get(str(grade).upper(), "⚪")
        name  = rec.get("product_name", "Unknown")
        ts    = rec.get("timestamp", "")[:19]
        status = str(rec.get("human_status", "")).upper()

        with st.expander(f"{icon} {name} — {grade} — {status} — {ts}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Defect %", da.get("defect_probability", "N/A"))
            c2.metric("Return %", qa.get("return_rate", "N/A"))
            c3.metric("Quality",  qa.get("quality_score", "N/A"))
            c4.metric("Risk",     da.get("risk_level", "N/A"))
            fb = rec.get("human_feedback", "")
            if fb:
                st.info(f"💬 Reviewer: {fb}")
            recs_list = reco.get("recommendations", [])
            if recs_list:
                st.markdown("**Recommendations:** " + " | ".join(str(r) for r in recs_list[:2]))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Feedback Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _compute_perf_from_files() -> dict:
    fb_dir  = Path(DATA_FEEDBACK_DIR)
    records = []
    if fb_dir.exists():
        for f in fb_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    records.append(json.load(fp))
            except Exception:
                pass

    total = max(len(records), 1)
    keyword_map = {
        "Product Analyzer": ["defect", "defective", "fault", "manufacturing"],
        "Quality Assessor": ["quality", "return", "rating", "score"],
        "Supply Chain":     ["supply", "supplier", "chain", "logistics"],
        "Cost Impact":      ["cost", "financial", "impact", "loss", "budget"],
        "Market Trend":     ["market", "demand", "trend", "competitive"],
    }
    counts = {agent: 0 for agent in keyword_map}
    for r in records:
        text = (r.get("feedback", "") or "").lower()
        if not text:
            continue
        for agent, keywords in keyword_map.items():
            if any(k in text for k in keywords):
                counts[agent] += 1

    return {
        agent: round(max(0.50, 0.85 - (count / total) * 0.35), 2)
        for agent, count in counts.items()
    }


def _compute_thresholds_from_files() -> dict:
    fb_dir  = Path(DATA_FEEDBACK_DIR)
    records = []
    if fb_dir.exists():
        for f in fb_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    records.append(json.load(fp))
            except Exception:
                pass

    total      = max(len(records), 1)
    rejections = sum(1 for r in records if str(r.get("decision", "")).upper() in ("REJECTED", "NEEDS_REVIEW"))
    rej_rate   = rejections / total
    rej_text   = " ".join(
        r.get("feedback", "") or ""
        for r in records
        if str(r.get("decision", "")).upper() in ("REJECTED", "NEEDS_REVIEW")
    ).lower()

    if rej_rate > 0.3 or "defect" in rej_text or "fault" in rej_text:
        rec_defect = max(5, int(15 * (1 - rej_rate * 0.3)))
        defect_adj = {
            "current":     15,
            "recommended": rec_defect,
            "reason":      f"Rejection rate {rej_rate:.0%} — defect threshold may be too permissive.",
        }
    else:
        defect_adj = {
            "current":     15,
            "recommended": 15,
            "reason":      "Current threshold appears appropriate based on feedback.",
        }

    if rej_rate > 0.3 or "return" in rej_text or "quality" in rej_text:
        rec_return = max(10, int(25 * (1 - rej_rate * 0.25)))
        return_adj = {
            "current":     25,
            "recommended": rec_return,
            "reason":      f"{rejections} rejection(s) suggest return rate threshold needs tightening.",
        }
    else:
        return_adj = {
            "current":     25,
            "recommended": 25,
            "reason":      "Current threshold appears appropriate based on feedback.",
        }

    return {"Defect Probability": defect_adj, "Return Rate": return_adj}


def display_ai_feedback_results(analysis: dict):
    if not analysis:
        st.info("No analysis results available.")
        return

    if st.session_state.get("feedback_stale"):
        st.warning("⚠ New feedback added since last analysis — re-run for fresh results.")

    status = analysis.get("analysis_status", "unknown")
    if status == "no_data":
        st.info("📭 No feedback data yet. Submit and save some assessments first.")
        return
    if status == "error":
        findings = analysis.get("key_findings") or ["Unknown error"]
        st.error(f"Analysis error: {findings[0]}")
        return

    findings = analysis.get("key_findings") or []
    recs     = analysis.get("recommendations") or []
    actions  = analysis.get("next_actions") or []
    mismatch = analysis.get("mismatch_count") or 0

    perf = _compute_perf_from_files()
    adj  = _compute_thresholds_from_files()

    c1, c2, c3 = st.columns(3)
    c1.metric("Key Findings",          len(findings))
    c2.metric("Prediction Mismatches", mismatch)
    c3.metric("Recommendations",       len(recs))
    st.divider()

    st.subheader("🔍 Key Findings")
    if findings:
        for f in findings:
            if f and str(f).strip() not in ("None", "null", ""):
                st.markdown(f"• {f}")
    else:
        st.info("No key findings available.")
    st.divider()

    st.subheader("🤖 Agent Performance")
    for agent_name, score in perf.items():
        try:
            v = float(score)
        except (TypeError, ValueError):
            continue
        col1, col2 = st.columns([4, 1])
        with col1:
            st.progress(min(max(v, 0.0), 1.0), text=agent_name)
        with col2:
            if v >= 0.80:
                st.success(f"{v:.0%}")
            elif v >= 0.65:
                st.warning(f"{v:.0%}")
            else:
                st.error(f"{v:.0%}")
    st.divider()

    st.subheader("⚙ Threshold Adjustments")
    for metric, info in adj.items():
        if not isinstance(info, dict):
            continue
        cur    = info.get("current",     "N/A")
        rec    = info.get("recommended", "N/A")
        reason = info.get("reason",      "")
        if str(cur) != str(rec):
            st.markdown(f"**{metric}**: `{cur}` → `{rec}` 🔽")
        else:
            st.markdown(f"**{metric}**: `{cur}` ✅ no change needed")
        if reason:
            st.caption(reason)
    st.divider()

    st.subheader("💡 Recommendations")
    if recs:
        for r in recs:
            if r and str(r).strip() not in ("None", "null"):
                st.markdown(f"• {r}")
    else:
        st.info("No recommendations available.")

    if actions:
        st.divider()
        st.subheader("➡ Next Actions")
        for a in actions:
            if a and str(a).strip() not in ("None", "null"):
                st.markdown(f"→ {a}")


def display_feedback_analysis():
    st.header("🔍 Feedback Analysis")

    fb_dir  = Path(DATA_FEEDBACK_DIR)
    records = []
    if fb_dir.exists():
        for f in fb_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    records.append(json.load(fp))
            except Exception:
                pass

    total      = len(records)
    rejections = sum(1 for r in records if str(r.get("decision", "")).upper() in ("REJECTED", "NEEDS_REVIEW"))
    approvals  = total - rejections
    rate       = rejections / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Feedback", total)
    c2.metric("Approvals",      approvals)
    c3.metric("Rejections",     rejections)
    c4.metric("Rejection Rate", f"{rate:.1%}")

    if total == 0:
        st.info("📭 No feedback yet. Run an assessment and save a decision first.")
        return

    st.divider()

    st.subheader("📋 Recent Feedback")
    rows = []
    for r in sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]:
        rows.append({
            "Product":  r.get("product_name", "Unknown"),
            "Decision": r.get("decision", "N/A"),
            "Grade":    r.get("final_quality_grade", "N/A"),
            "Defect %": r.get("defect_probability", "N/A"),
            "Return %": r.get("return_rate", "N/A"),
            "Comment":  (r.get("feedback", "") or "")[:60],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("🤖 AI-Powered Feedback Analysis")
    st.caption("Analyses rejection patterns, agent performance, and recommends threshold adjustments.")

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("▶ Run AI Feedback Analysis", type="primary"):
            with st.spinner("Analysing feedback patterns with AI agents..."):
                try:
                    llm    = initialize_gemini_llm()
                    flow   = QualityAssessmentFlow(llm)
                    result = flow.analyze_feedback()
                    st.session_state["ai_feedback_cache"] = result
                    st.session_state["feedback_stale"]    = False
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
    with col2:
        if st.button("🗑 Clear Results"):
            st.session_state["ai_feedback_cache"] = None
            st.rerun()

    if st.session_state.get("ai_feedback_cache"):
        st.divider()
        display_ai_feedback_results(st.session_state["ai_feedback_cache"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _eval_model(model_pkl, scaler_pkl, encoder_pkl, eval_csv, target_col):
    root = Path(__file__).parent
    try:
        model   = pickle.load(open(root / "ml" / "models" / model_pkl,   "rb"))
        scaler  = pickle.load(open(root / "ml" / "models" / scaler_pkl,  "rb"))
        encoder = pickle.load(open(root / "ml" / "models" / encoder_pkl, "rb"))
        df = pd.read_csv(root / "data" / "evaluation_dataset" / eval_csv)
        df["product_manufacturing_country"] = df["product_manufacturing_country"].str.upper().str.strip()

        fallbacks = {
            "defect_probability": ["defect_probability", "predicted_defect_probability_percent"],
            "return_rate":        ["return_rate", "predicted_return_rate_percent"],
        }
        actual_col = next(
            (c for c in fallbacks.get(target_col, [target_col]) if c in df.columns), None
        )
        if actual_col is None:
            return {"error": f"Column '{target_col}' not found. Available: {list(df.columns)}"}

        X = df[ML_FEATURES_ORDERED].copy()
        X["product_manufacturing_country"] = encoder.transform(X["product_manufacturing_country"])
        X_s    = scaler.transform(X)
        y_true = df[actual_col].values
        y_pred = np.clip(model.predict(X_s), 0, 100)
        mse    = float(mean_squared_error(y_true, y_pred))
        mae    = float(mean_absolute_error(y_true, y_pred))
        r2     = float(r2_score(y_true, y_pred))
        rmse   = float(np.sqrt(mse))
        return {"R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MSE": round(mse, 4)}
    except Exception as e:
        return {"error": str(e)}


def display_model_evaluation_tab():
    st.header("📐 Model Evaluation")
    st.info("Runs both ML models against the evaluation dataset and shows real metrics.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Defect Classifier")
        if st.button("Evaluate Defect Classifier"):
            with st.spinner("Evaluating..."):
                st.session_state["eval_defect_metrics"] = _eval_model(
                    "defect_classifier.pkl", "defect_scaler.pkl", "defect_encoder.pkl",
                    "defect_classifier_evaluation.csv", "defect_probability",
                )
        m = st.session_state.get("eval_defect_metrics")
        if m:
            if "error" in m:
                st.error(m["error"])
            else:
                st.metric("R² Score", m["R2"])
                st.metric("RMSE",     m["RMSE"])
                st.metric("MAE",      m["MAE"])
                st.metric("MSE",      m["MSE"])
                st.success("Higher R² = better model fit (1.0 is perfect)")

    with col2:
        st.subheader("🔄 Return Rate Predictor")
        if st.button("Evaluate Return Rate Predictor"):
            with st.spinner("Evaluating..."):
                st.session_state["eval_return_metrics"] = _eval_model(
                    "return_predictor.pkl", "return_scaler.pkl", "return_encoder.pkl",
                    "return_rate_predictor_evaluation.csv", "return_rate",
                )
        m = st.session_state.get("eval_return_metrics")
        if m:
            if "error" in m:
                st.error(m["error"])
            else:
                st.metric("R² Score", m["R2"])
                st.metric("RMSE",     m["RMSE"])
                st.metric("MAE",      m["MAE"])
                st.metric("MSE",      m["MSE"])
                st.success("Higher R² = better model fit (1.0 is perfect)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="QualityGuard", page_icon="🛡️", layout="wide")
    check_models_exist()
    initialize_session()

    st.title("🛡️ QualityGuard — AI Product Quality Assessment")
    st.caption("Multi-agent AI system built by The Crew🧑‍💻 with ❤️")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🆕 New Assessment",
        "📋 Assessment History",
        "🔍 Feedback Analysis",
        "📐 Model Evaluation",
    ])

    with tab1:
        st.header("New Quality Assessment")
        try:
            products = load_sample_products()
        except Exception as e:
            st.error(f"Could not load products: {e}")
            return

        names         = [p["name"] for p in products]
        selected_name = st.selectbox("Select product:", names)
        product_data  = next(p for p in products if p["name"] == selected_name)

        # Auto-clear old assessment when product changes
        if st.session_state.get("last_selected_product") != selected_name:
            st.session_state["current_assessment"]    = None
            st.session_state["last_selected_product"] = selected_name

        with st.expander("📦 Product Details", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price",      f"${product_data['product_price_usd']:.2f}")
            c2.metric("Rating",     f"{product_data['customer_average_rating']}/5")
            c3.metric("Reviews",    product_data["customer_review_count"])
            c4.metric("Country",    product_data["product_manufacturing_country"])
            c5.metric("Demand Idx", product_data["market_demand_index"])

        running = st.session_state.get("assessment_running", False)
        if st.button("🚀 Start Quality Assessment", type="primary", disabled=running):
            st.session_state["assessment_running"] = True
            status_box = st.empty()
            progress   = st.progress(0)

            try:
                progress.progress(15)
                status_box.info("🔍 Running multi-agent analysis...")

                llm    = initialize_gemini_llm()
                flow   = QualityAssessmentFlow(llm)
                result = flow.kickoff(product_data)

                progress.progress(100)
                status_box.success("🎉 Assessment complete!")

                result["assessment_id"] = str(uuid.uuid4())[:8]
                st.session_state["current_assessment"] = result
                st.session_state["human_decision"]     = None

            except Exception as e:
                status_box.empty()
                progress.empty()
                st.error(f"Assessment failed: {e}")
                st.exception(e)
            finally:
                st.session_state["assessment_running"] = False
        if running:
            import streamlit.components.v1 as components

            components.html("""
                <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: transparent; }
                
                .row {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 14px 18px;
                    border: 0.5px solid rgba(0,0,0,0.1);
                    border-radius: 12px;
                    background: white;
                }
                
                @media (prefers-color-scheme: dark) {
                    .row { background: #1e1e1e; border-color: rgba(255,255,255,0.1); }
                    .msg { color: #ccc; }
                }
                
                .dots {
                    display: flex;
                    gap: 5px;
                    flex-shrink: 0;
                }
                
                .dot {
                    width: 7px;
                    height: 7px;
                    border-radius: 50%;
                    background: #7F77DD;
                    animation: pulse 1.4s ease-in-out infinite;
                }                
                
                .dot:nth-child(1) { animation-delay: 0s; }
                .dot:nth-child(2) { animation-delay: 0.2s; }
                .dot:nth-child(3) { animation-delay: 0.4s; }

                @keyframes pulse {
                    0%, 80%, 100% { opacity: 0.25; transform: scale(0.8); }
                    40% { opacity: 1; transform: scale(1.2); }
                }

                .msg {
                    font-size: 14px;
                    color: #555;
                    overflow: hidden;
                    white-space: nowrap;
                    border-right: 2px solid #7F77DD;
                    animation: type 4s steps(50) infinite, blink 0.6s step-end infinite alternate;
                }
                
                @keyframes type {
                    0% { width: 0; }
                    70% { width: 100%; }
                    100% { width: 100%; }
                }
                
                @keyframes blink {
                    from { border-color: #7F77DD; }
                    to { border-color: transparent; }
                }
                </style>
                
                <div class="row">
                    <div class="dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                    <div class="msg" id="msg">Analysing product quality with AI agents...</div>
                </div>
                
                <script>
                var msgs = [
                    "Analysing product quality with AI agents...",
                    "Running machine learning predictions...",
                    "Evaluating supply chain and market data...",
                    "Generating quality assessment report..."
                ];
                var i = 0;
                setInterval(function() {
                    i = (i + 1) % msgs.length;
                    document.getElementById('msg').textContent = msgs[i];
                }, 4000);
                </script>
                """, height=70)
                
            st.info("⏳ Assessment in progress — please wait...")
        

        if st.session_state["current_assessment"]:
            results = st.session_state["current_assessment"]
            display_assessment_results(results)

            st.divider()
            st.subheader("👤 Human Review Decision")
            decision = st.radio(
                "Your decision:",
                [STATUS_APPROVED, STATUS_REJECTED, STATUS_PENDING_REVIEW],
                horizontal=True,
            )
            feedback = st.text_area(
                "Comments (optional):",
                placeholder="Add notes about why you approve/reject this product...",
            )
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("💾 Save Decision", type="primary"):
                    save_assessment(results, decision, feedback)
                    st.success(f"Decision '{decision}' saved!")
                    st.session_state["current_assessment"] = None
                    st.rerun()

    with tab2:
        display_assessment_history()
    with tab3:
        display_feedback_analysis()
    with tab4:
        display_model_evaluation_tab()


if __name__ == "__main__":
    main()
