import os
import tempfile
import time
import base64
import pickle
import re
from difflib import SequenceMatcher
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set matplotlib default font to Poppins (if available)
plt.rcParams['font.family'] = "Poppins, sans-serif"

# ----------------- Configuration -----------------
PROPHET_DATA = "monthly_dataset.csv"
PROPHET_MODEL_DIR = "Prophet_models"
UPCOMING_MONTHLY_COPY = "salesmonthly_updated_data.csv"
XGB_DATA = "weekly_dataset.csv"
UPCOMING_WEEKLY_COPY = "salesweekly_updated_data.csv"

MEDICINE_CATEGORIES = {
    'M01AB': 'M01AB Drug Type',
    'M01AE': 'M01AE Drug Type',
    'N02BA': 'N02BA Drug Type',
    'N02BE': 'N02BE Drug Type',
    'N05B': 'N05B Drug Type',
    'N05C': 'N05C Drug Type',
    'R03': 'R03 Drug Type',
    'R06': 'R06 Drug Type'
}

THEME_BLUE = "#2888bc"
THEME_PURPLE = "#3b2259"
CANCEL_RED = "#e74c3c"

# ----------------- Gemini API key (user-provided) -----------------
# NOTE: environment variable; keep as-is if provided
os.environ.setdefault("GEMINI_API_KEY", "AIzaSyC1bl7-pGAEh98YuozBf52AFriizMpsfL0")

# ----------------- Utilities -----------------
def safe_save_csv(df_obj: pd.DataFrame, file_path: str, max_retries: int = 2, retry_delay: float = 0.5):
    dirpath = os.path.dirname(os.path.abspath(file_path)) or "."
    os.makedirs(dirpath, exist_ok=True)
    for attempt in range(max_retries + 1):
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_csv_", suffix=".csv", dir=dirpath)
        os.close(fd)
        try:
            df_obj.to_csv(tmp_path, index=False)
            os.replace(tmp_path, file_path)
            return
        except PermissionError:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

def check_file_exists(path):
    return os.path.isfile(path)

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

def simple_fuzzy_pick(description: str, categories: list):
    if not categories:
        return None, 0.0
    best, best_score = None, -1.0
    for cat in categories:
        score = similarity(description, cat)
        if score > best_score:
            best_score = score
            best = cat
    return best, float(best_score)

def detect_frequency(df_local):
    if "datum" not in df_local.columns or df_local["datum"].dropna().empty:
        return "daily"
    df_local["datum"] = pd.to_datetime(df_local["datum"], errors="coerce")
    diffs = df_local["datum"].diff().dropna().dt.days
    if diffs.empty:
        return "daily"
    avg_gap = diffs.mean()
    if avg_gap >= 25:
        return "monthly"
    elif avg_gap >= 5:
        return "weekly"
    else:
        return "daily"

@st.cache_data(show_spinner=False)
def load_monthly_data(path=PROPHET_DATA):
    if not os.path.isfile(path):
        return pd.DataFrame(columns=["datum"])
    df = pd.read_csv(path)
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_weekly_data(path=XGB_DATA):
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        header = []
    parse_dates = ["datum"] if "datum" in header else None
    df = pd.read_csv(path, parse_dates=parse_dates)
    if "datum" in df.columns:
        df.set_index("datum", inplace=True)
    return df

def prepare_data_xgb(df, categories, lags=4):
    if not categories:
        return pd.DataFrame()
    existing_cats = [c for c in categories if c in df.columns]
    if not existing_cats:
        return pd.DataFrame()
    df_prep = df[existing_cats].copy()
    if not isinstance(df_prep.index, pd.DatetimeIndex):
        df_prep.index = pd.to_datetime(df_prep.index)
    df_prep = df_prep.sort_index()
    df_prep['year'] = df_prep.index.year
    df_prep['month'] = df_prep.index.month
    try:
        df_prep['week'] = df_prep.index.isocalendar().week
    except Exception:
        df_prep['week'] = df_prep.index.to_series().dt.isocalendar().week
    for col in existing_cats:
        for l in range(1, lags + 1):
            df_prep[f'{col}_lag{l}'] = df_prep[col].shift(l)
    df_prep = df_prep.dropna()
    return df_prep

def train_xgb(X, y, prev_model=None):
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.1,
        'max_depth': 6,
        'eval_metric': 'rmse'
    }
    num_round = 100
    if prev_model:
        bst = xgb.train(params, dtrain, num_round, xgb_model=prev_model)
    else:
        bst = xgb.train(params, dtrain, num_round)
    return bst

def forecast_col_xgb(col, n, model, df, lags=4, freq='W'):
    preds = []
    recent = list(df[col].tail(lags))
    if len(recent) < lags:
        if recent:
            pad_val = recent[0]
            recent = [pad_val] * (lags - len(recent)) + recent
        else:
            recent = [0.0] * lags
    current_lags = recent.copy()
    current_date = df.index.max()
    for i in range(n):
        if freq == 'W':
            next_date = current_date + pd.Timedelta(weeks=1)
        else:
            next_date = current_date + pd.Timedelta(days=1)
        feat_dict = {
            'year': next_date.year,
            'month': next_date.month,
            'week': next_date.isocalendar()[1],
        }
        for l in range(1, lags + 1):
            feat_dict[f'{col}_lag{l}'] = current_lags[-l]
        X_pred = pd.DataFrame([feat_dict])
        try:
            pred = float(model.predict(xgb.DMatrix(X_pred))[0])
        except Exception:
            pred = 0.0
        preds.append(pred)
        current_lags = current_lags[1:] + [pred]
        current_date = next_date
    if freq == 'W':
        future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(weeks=1), periods=n, freq='W')
    else:
        future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=n, freq='D')
    return pd.Series(preds, index=future_dates)

# ----------------- Stock update helper -----------------
def update_upcoming_file(file_path, source_df, new_tablet, new_volume, update_date):
    if not os.path.isfile(file_path):
        base = source_df.copy() if source_df is not None and not source_df.empty else pd.DataFrame(columns=['datum'])
        safe_save_csv(base, file_path)

    try:
        df_local = pd.read_csv(file_path)
    except Exception:
        df_local = pd.DataFrame(columns=['datum'])

    if "datum" in df_local.columns:
        df_local["datum"] = pd.to_datetime(df_local["datum"], errors="coerce")
    else:
        df_local["datum"] = pd.to_datetime([])

    categories = [c for c in df_local.columns if c != "datum"]
    if not categories:
        if source_df is not None and not source_df.empty:
            categories = [c for c in source_df.columns if c != "datum"]
            for c in categories:
                df_local[c] = 0
        else:
            df_local[new_tablet] = 0
            categories = [new_tablet]

    detected_cat, _ = simple_fuzzy_pick(new_tablet, categories)
    chosen_category = detected_cat if detected_cat in categories else categories[0]

    if chosen_category not in df_local.columns:
        df_local[chosen_category] = 0

    freq_type = detect_frequency(df_local)
    today = pd.to_datetime(update_date)

    if freq_type == "monthly":
        mask = df_local["datum"].dt.to_period("M") == today.to_period("M")
    elif freq_type == "weekly":
        mask = (df_local["datum"].dt.isocalendar().week == today.isocalendar().week) & (df_local["datum"].dt.isocalendar().year == today.isocalendar().year)
    else:
        mask = df_local["datum"].dt.date == today.date()

    if mask.any():
        idx = df_local[mask].index[0]
        existing = pd.to_numeric(df_local.at[idx, chosen_category], errors="coerce")
        if np.isnan(existing):
            existing = 0
        df_local.at[idx, chosen_category] = existing + new_volume
        action = "updated"
    else:
        if "datum" not in df_local.columns:
            df_local["datum"] = pd.to_datetime([])
        new_row = {col: 0 for col in df_local.columns}
        new_row["datum"] = today
        if chosen_category not in new_row:
            df_local[chosen_category] = 0
            new_row[chosen_category] = new_volume
        else:
            new_row[chosen_category] = new_volume
        df_local.loc[len(df_local)] = new_row
        action = "created"

    safe_save_csv(df_local, file_path)
    return action, chosen_category, df_local

# ----------------- Preview Styler (blue Excel-like selection) -----------------
def style_preview_dataframe(df: pd.DataFrame, highlight: dict = None):
    if df is None or df.empty:
        return df

    df_local = df.copy()
    if "datum" in df_local.columns:
        df_local["datum"] = pd.to_datetime(df_local["datum"], errors="coerce").dt.date
    else:
        return df_local.style

    if not highlight or "date" not in highlight or "col" not in highlight:
        return df_local.style

    try:
        highlight_date = pd.to_datetime(highlight.get("date")).date()
    except Exception:
        highlight_date = None
    highlight_col = highlight.get("col")

    def _row_style(row):
        styles = []
        for col in df_local.columns:
            css = ""
            if col == highlight_col:
                row_date = row.get("datum", None)
                if row_date == highlight_date:
                    css = (
                        "background-color: #e74c3c; "
                        "color: #ffffff; "
                        "font-weight: 700; "
                        "box-shadow: inset 0 -2px 0 rgba(0,0,0,0.12); "
                        "border: 2px solid #c0392b; "
                        "border-radius: 4px; "
                        "padding: 4px;"
                    )
            styles.append(css)
        return styles

    styler = df_local.style.apply(_row_style, axis=1)
    styler = styler.set_table_attributes('class="dataframe-preview" style="width:100%; border-collapse:collapse;"')
    return styler

# ----------------- Preview HTML builder (fallback) -----------------
def build_preview_html(df: pd.DataFrame, title: str, highlight: dict = None, max_rows: int = 2000):
    # Include Google Fonts link here so iframe/HTML preview sees Poppins
    if df is None or df.empty:
        table_html = "<p>No data to preview.</p>"
        notice = ""
    else:
        df_to_show = df.copy()
        if "datum" in df_to_show.columns:
            df_to_show["datum"] = pd.to_datetime(df_to_show["datum"], errors="coerce")
        total_rows = len(df_to_show)
        if max_rows and total_rows > max_rows:
            df_to_show = df_to_show.tail(max_rows)
            notice = f"<p>Showing last {max_rows} rows (of {total_rows}).</p>"
        else:
            notice = f"<p>Rows: {total_rows}</p>"

        headers = list(df_to_show.columns)
        highlight_date = None
        highlight_col = None
        if highlight:
            try:
                if highlight.get("date") is not None:
                    highlight_date = pd.to_datetime(highlight["date"]).date()
            except Exception:
                highlight_date = None
            highlight_col = highlight.get("col")

        ths = "".join([f"<th>{h}</th>" for h in headers])
        trs = ""
        for _, r in df_to_show.iterrows():
            tr_cells = ""
            row_date = None
            if "datum" in df_to_show.columns:
                try:
                    row_date = pd.to_datetime(r["datum"]).date()
                except Exception:
                    row_date = None
            for h in headers:
                cell_val = r[h]
                cell_text = "" if pd.isna(cell_val) else str(cell_val)
                cell_style = ""
                if highlight_date is not None and highlight_col is not None and row_date == highlight_date and h == highlight_col:
                    cell_style = f"background:#e74c3c;color:#fff;border:2px solid #c0392b;padding:4px;border-radius:4px;"
                elif highlight_date is not None and row_date == highlight_date:
                    cell_style = "background:#f4f6f8;"
                tr_cells += f'<td style="{cell_style}">{cell_text}</td>'
            trs += f"<tr>{tr_cells}</tr>"

        table_html = f"""
        <table class="preview-table">
          <thead><tr>{ths}</tr></thead>
          <tbody>{trs}</tbody>
        </table>
        """

    full_html = f"""
    <html>
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body {{ margin:0; font-family: 'Poppins', Arial, Helvetica, sans-serif; }}
          table.preview-table {{ border-collapse: collapse; width:100%; font-size:13px; font-family: 'Poppins', sans-serif; }}
          table.preview-table th, table.preview-table td {{ padding:6px 8px; border:1px solid #e6e6e6; text-align:left; }}
          table.preview-table thead th {{ background:#f4f6f8; position: sticky; top: 0; z-index:2; }}
        </style>
      </head>
      <body>
        <div style="padding:12px;">
          <h3 style="margin:0;padding:0;font-family: 'Poppins', sans-serif;">{title}</h3>
          {notice}
          {table_html}
        </div>
      </body>
    </html>
    """
    return full_html

# ----------------- Blob-based Confirmation Modal -----------------
def show_confirmation_modal(payload: dict, theme_blue: str = THEME_BLUE, cancel_red: str = CANCEL_RED, height: int = 700):
    b = payload.get("bytes", b"")
    fname = payload.get("name", "updated.csv")
    try:
        b64 = base64.b64encode(b).decode()
    except Exception:
        b64 = ""

    # Include Google Fonts link inside modal HTML so it renders in iframe
    html = f"""
    <!doctype html>
    <html>
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
          :root {{
            --blue: {theme_blue};
            --red: {cancel_red};
          }}
          html,body{{margin:0;padding:0;height:100%;font-family:'Poppins', Inter, Arial, Helvetica, sans-serif;}}
          .overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.45);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 999999;
            -webkit-font-smoothing:antialiased;
          }}
          .modal {{
            width: 540px;
            max-width: calc(100% - 32px);
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 12px 40px rgba(10, 10, 10, 0.3);
            overflow: hidden;
            animation: pop 160ms ease-out;
            font-family: 'Poppins', sans-serif;
          }}
          @keyframes pop {{
            from {{ transform: translateY(8px) scale(.995); opacity:0; }}
            to {{ transform: translateY(0) scale(1); opacity:1; }}
          }}
          .modal-header {{
            padding: 18px 20px;
            background: linear-gradient(90deg, var(--blue), {THEME_PURPLE});
            color: #fff;
            display:flex;
            align-items:center;
            justify-content:space-between;
          }}
          .modal-header h3 {{ margin:0; font-size:18px; font-weight:700; font-family: 'Poppins', sans-serif; }}
          .modal-body {{ padding: 18px 20px; color: #111827; background:#fff; font-family: 'Poppins', sans-serif; }}
          .modal-actions {{
            display:flex;
            gap:12px;
            justify-content:flex-end;
            padding: 16px 18px;
            background:#fff;
          }}
          .btn {{
            padding: 10px 16px;
            border-radius: 8px;
            font-weight:700;
            text-decoration:none;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            cursor:pointer;
            border: none;
            font-size:14px;
            font-family: 'Poppins', sans-serif;
          }}
          .btn-download {{
            background: var(--blue);
            color: white;
            border: 1px solid var(--blue);
            box-shadow: 0 6px 18px rgba(40,136,188,0.18);
          }}
          .btn-cancel {{
            background: var(--red);
            color: #fff;
            border: 1px solid rgba(0,0,0,0.06);
          }}
        </style>
      </head>
      <body>
        <div class="overlay" id="overlay" tabindex="0" role="dialog" aria-modal="true" aria-label="Download confirmation">
          <div class="modal" role="document" id="modal">
            <div class="modal-header">
              <h3 id="modal-title">Confirm Download</h3>
              <button id="close" aria-label="Close" style="background:transparent;border:none;color:rgba(255,255,255,0.95);font-size:20px;font-weight:700;cursor:pointer;">✕</button>
            </div>
            <div class="modal-body">
              <p>Are you sure you want to download the updated CSV?</p>
              <div style="background:#f8fafc;border:1px solid #e6eef6;padding:10px 12px;border-radius:8px;font-size:14px;color:#0f172a;font-family:'Poppins',sans-serif;">
                <strong>File:</strong> {fname} &nbsp;&nbsp; <strong>Type:</strong> CSV
              </div>
            </div>
            <div class="modal-actions">
              <button class="btn btn-cancel" id="cancel-btn">Cancel</button>
              <button class="btn btn-download" id="download-btn">Download</button>
            </div>
          </div>
        </div>

        <script>
          (function() {{
            const b64 = "{b64}";
            const fname = "{fname}";
            let downloadUrl = null;

            function base64ToBlobUrl(b64data, mime) {{
              try {{
                const byteString = atob(b64data);
                const len = byteString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {{
                  bytes[i] = byteString.charCodeAt(i);
                }}
                const blob = new Blob([bytes], {{ type: mime }});
                if (downloadUrl) {{
                  URL.revokeObjectURL(downloadUrl);
                }}
                downloadUrl = URL.createObjectURL(blob);
                return downloadUrl;
              }} catch (e) {{
                console.error("Failed to create blob URL:", e);
                return null;
              }}
            }}

            const overlay = document.getElementById('overlay');
            const modal = document.getElementById('modal');
            const closeBtn = document.getElementById('close');
            const cancelBtn = document.getElementById('cancel-btn');
            const downloadBtn = document.getElementById('download-btn');

            function hideModal() {{
              overlay.style.display = 'none';
              if (downloadUrl) {{
                setTimeout(() => {{ URL.revokeObjectURL(downloadUrl); downloadUrl = null; }}, 2000);
              }}
            }}

            overlay.addEventListener('click', function(e) {{
              if (e.target === overlay) {{
                hideModal();
              }}
            }});
            closeBtn.addEventListener('click', hideModal);
            cancelBtn.addEventListener('click', hideModal);
            document.addEventListener('keydown', function(e) {{
              if (e.key === 'Escape') hideModal();
            }});

            downloadBtn.addEventListener('click', function(e) {{
              e.preventDefault();
              e.stopPropagation();
              
              const url = base64ToBlobUrl(b64, 'text/csv;charset=utf-8;');
              if (!url) {{
                alert("Download failed: unable to prepare file for download.");
                return;
              }}
              
              try {{
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = fname;
                a.rel = 'noopener';
                a.target = '_blank';
                document.body.appendChild(a);
                a.click();
                setTimeout(() => {{
                  document.body.removeChild(a);
                  hideModal();
                }}, 100);
              }} catch (err) {{
                console.error("Download error:", err);
                alert("Download failed. Please try again.");
              }}
            }});

            setTimeout(() => {{ modal.focus && modal.focus(); }}, 50);
          }})();
        </script>
      </body>
    </html>
    """
    components.html(html, height=height, scrolling=False, width="100%", sandbox=["allow-scripts", "allow-downloads", "allow-popups"])

# ----------------- Gemini helpers (attempt call, fallback to deterministic local) -----------------
def build_trend_summary(series: pd.Series):
    s = series.dropna().astype(float)
    if s.empty:
        return {"start": 0.0, "end": 0.0, "mean": 0.0, "std": 0.0, "pct_change": 0.0, "n_points": 0, "recent_slope": 0.0}
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    mean = float(s.mean())
    std = float(s.std(ddof=0)) if len(s) > 1 else 0.0
    pct_change = ((end - start) / (start if start != 0 else 1)) * 100
    n = len(s)
    k = min(5, n)
    if k >= 2:
        y = s.iloc[-k:].values
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        recent_slope = float(m)
    else:
        recent_slope = 0.0
    return {"start": start, "end": end, "mean": mean, "std": std, "pct_change": pct_change, "n_points": n, "recent_slope": recent_slope}

def build_gemini_prompt(product_name: str, timeframe: str, summary: dict):
    prompt = (
        f"I have a forecast for product '{product_name}' covering {timeframe}.\n\n"
        f"Numeric summary:\n"
        f"- Start value: {summary['start']:.2f}\n"
        f"- End value: {summary['end']:.2f}\n"
        f"- Percent change (start->end): {summary['pct_change']:.2f}%\n"
        f"- Mean forecast: {summary['mean']:.2f}, Std dev: {summary['std']:.2f}\n"
        f"- Data points: {summary['n_points']}\n"
        f"- Recent slope (units per period): {summary['recent_slope']:.4f}\n\n"
        "Write a single plain-English paragraph (4 sentences) that a non-technical person can understand. "
        "Mention the key numbers (percent change and timeframe), describe whether sales are increasing, decreasing, or stable, "
        "and include one short practical takeaway. Keep it simple and concrete (e.g., 'sales are estimated to rise by 34% after one month'). "
        "Return only the paragraph, without bullets, headers, or extra metadata."
    )
    return prompt

def call_gemini_text(prompt: str, model_env: str = None, max_output_tokens: int = 512):
    try:
        api_key = os.getenv("GEMINI_API_KEY", None)
        model_name = model_env or os.getenv("GEMINI_MODEL", None) or "gemini-1.0"
        try:
            import google.generativeai as genai
        except Exception:
            return None
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass
        try:
            resp = None
            if hasattr(genai, "generate"):
                try:
                    resp = genai.generate(model=model_name, prompt=prompt, max_output_tokens=max_output_tokens)
                except TypeError:
                    resp = genai.generate(model=model_name, input=prompt, max_output_tokens=max_output_tokens)
            elif hasattr(genai, "chat"):
                resp = genai.chat.create(model=model_name, messages=[{"role":"user","content":prompt}], max_output_tokens=max_output_tokens)
            else:
                return None
            if resp is None:
                return None
            if isinstance(resp, dict):
                if "candidates" in resp and resp["candidates"]:
                    c0 = resp["candidates"][0]
                    if isinstance(c0, dict) and "content" in c0:
                        return c0["content"]
                    if isinstance(c0, dict) and "message" in c0:
                        return c0["message"].get("content", "")
                    if isinstance(c0, dict) and "text" in c0:
                        return c0["text"]
                if "output" in resp and resp["output"]:
                    segments = resp["output"]
                    text_parts = []
                    for seg in segments:
                        if isinstance(seg, dict):
                            text_parts.append(seg.get("content", "") or seg.get("text", ""))
                        else:
                            text_parts.append(str(seg))
                    return "".join(text_parts).strip() or None
                return str(resp)
            text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
            if text:
                return text
            try:
                candidates = getattr(resp, "candidates", None)
                if candidates and len(candidates) > 0:
                    cand = candidates[0]
                    txt = getattr(cand, "content", None) or getattr(cand, "message", None)
                    if isinstance(txt, dict):
                        return txt.get("content", "")
                    if isinstance(txt, str):
                        return txt
            except Exception:
                pass
            return str(resp)
        except Exception:
            return None
    except Exception:
        return None

def build_local_explanation(product_name: str, timeframe: str, summary: dict):
    pct = summary.get("pct_change", 0.0)
    start = summary.get("start", 0.0)
    end = summary.get("end", 0.0)
    mean = summary.get("mean", 0.0)
    std = summary.get("std", 0.0)
    n = summary.get("n_points", 0)
    slope = summary.get("recent_slope", 0.0)

    s1 = f"For {product_name}, the {timeframe} forecast shows sales moving from about {start:.1f} to {end:.1f}, a change of {pct:.1f}%."
    if pct >= 2:
        trend_word = "increasing"
    elif pct <= -2:
        trend_word = "decreasing"
    else:
        trend_word = "stable"
    s2 = f"Overall the trend appears {trend_word} over the period, with an average forecast value around {mean:.1f} and typical variation (std dev) of {std:.1f}."
    s3 = f"The forecast uses {n} modelled points and recent short-term slope suggests a change of about {slope:.2f} units per period."
    s4 = f"In simple terms, you can expect roughly a {pct:.1f}% change in sales across this {timeframe}; plan inventory and promotions accordingly."
    return " ".join([s1, s2, s3, s4])

def _split_into_sentences(text: str):
    if not text or not isinstance(text, str):
        return []
    cleaned = re.sub(r'^[\s]*[\-\*\u2022]\s*', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*\n\s*', ' ', cleaned)
    cleaned = cleaned.strip()
    parts = re.split(r'(?<=[.!?])\s+', cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def get_forced_explanation(product_name: str, timeframe: str, summary: dict, n_sentences: int = 4):
    prompt = build_gemini_prompt(product_name=product_name, timeframe=timeframe, summary=summary)
    gemini_text = call_gemini_text(prompt)
    gemini_sentences = _split_into_sentences(gemini_text) if gemini_text else []
    local_para = build_local_explanation(product_name=product_name, timeframe=timeframe, summary=summary)
    local_sentences = _split_into_sentences(local_para)

    final_sentences = []
    if gemini_sentences:
        final_sentences.extend(gemini_sentences)
    if len(final_sentences) < n_sentences:
        for s in local_sentences:
            if len(final_sentences) >= n_sentences:
                break
            if s not in final_sentences:
                final_sentences.append(s)
    if len(final_sentences) < n_sentences:
        while len(final_sentences) < n_sentences:
            final_sentences.append(local_sentences[-1] if local_sentences else "")
    final_sentences = final_sentences[:n_sentences]
    paragraph = " ".join([s.rstrip(" .") + ('.' if not s.endswith(('.', '?', '!')) else '') for s in final_sentences])
    paragraph = re.sub(r'\s+', ' ', paragraph).strip()
    return paragraph

# ----------------- Streamlit UI -----------------
st.set_page_config(
    page_title="Pharma Forecast (Prophet + XGBoost)", 
    layout="wide",
    page_icon="pharmacy.png"
)

# ----------------- Global Poppins font injection & force override -----------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      /* 1) Most global containers (Streamlit changes classes often) */
      html, body, .stApp, .block-container, .reportview-container, .main, .element-container, .css-1y4p8pa, [data-testid="stSidebar"] {
        font-family: 'Poppins', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial !important;
      }
      /* 2) Target many generated class patterns commonly used by Streamlit */
      [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
      }
      [class^="css"] {
        font-family: 'Poppins', sans-serif !important;
      }
      /* 3) Widgets, inputs, buttons, selects, markdown, dataframes */
      .stButton>button, .stDownloadButton>button, button, input, textarea, select, .stTextInput, .stNumberInput, .stDateInput, .stSelectbox, .stRadio {
        font-family: 'Poppins', sans-serif !important;
      }
      .stMarkdown, .stText, .stLabel, .stMetric, .stMetricValue, .stMetricLabel {
        font-family: 'Poppins', sans-serif !important;
      }
      /* 4) Pandas/preview tables and components */
      .dataframe-preview, .preview-table, table, th, td, thead, tbody {
        font-family: 'Poppins', sans-serif !important;
      }
      /* 5) Plotly (canvas/SVG) general hint */
      .js-plotly-plot, .plotly, .plot-container {
        font-family: 'Poppins', sans-serif !important;
      }
      /* 6) Ensure heavy weights for headings */
      h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif !important; font-weight:600 !important; }
      /* 7) Small utility tweaks */
      .stButton>button { font-weight:600 !important; border-radius:8px !important; }
      .stDownloadButton>button { font-weight:600 !important; border-radius:8px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title + CSS
st.markdown(f"""
    <h1 style='text-align: center;
               background: linear-gradient(135deg, {THEME_BLUE} 0%, {THEME_PURPLE} 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 800;
               font-family: "Poppins", sans-serif;'>
        Pharmaceutical Sales Forecast Dashboard
    </h1>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .metric-box {{
      background: linear-gradient(135deg, {THEME_BLUE} 0%, {THEME_PURPLE} 100%);
      border-radius:10px;
      padding:12px;
      text-align:center;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
      transition: all 0.3s ease;
      transform: translateY(0);
      margin-bottom: 8px;
      font-family: 'Poppins', sans-serif !important;
    }}
    .metric-box:hover {{
      transform: translateY(-5px);
      box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }}
    .metric-label {{ font-size:14px; color:#FFFFFF; margin-bottom:6px; font-weight:500; font-family:'Poppins',sans-serif !important; }}
    .metric-value {{ font-size:22px; font-weight:700; color:#FFFFFF; font-family:'Poppins',sans-serif !important; }}
    [data-testid="stSidebar"] {{background-color: {THEME_BLUE};}}
    .primary-blue .stButton>button {{
        background: #000000 !important; color: #000 !important; border: 1px solid {THEME_BLUE};
        font-weight: 700; border-radius: 8px;
    }}
    /* Dataframe preview styling for blue selection */
    .dataframe-preview table {{ width: 100%; border-collapse: collapse; font-family: 'Poppins', sans-serif !important; }}
    .dataframe-preview th, .dataframe-preview td {{ border: 1px solid #e6e6e6; padding: 6px 8px; text-align: left; vertical-align: middle; font-family: 'Poppins', sans-serif !important; }}
    .dataframe-preview thead th {{ background: #f4f6f8; position: sticky; top: 0; z-index: 2; }}
    .dataframe-preview td {{ color: inherit !important; }}
    .dataframe-preview td[style*="border: 2px solid"] {{
        overflow: visible;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar controls (REPLACEMENT) ----------------
# Inject sidebar-specific CSS for larger white labels & clearer controls
st.markdown(
    """
    <style>
    /* Sidebar background already set, force white text for readability */
    [data-testid="stSidebar"] {
      color: #000000 !important;
      padding-top: 18px;
    }

    /* Custom label styles inserted above each widget */
    .sidebar-label {
      font-family: 'Poppins', sans-serif;
      font-size: 16px;
      font-weight: 800;
      color: #ffffff;
      margin: 6px 0 6px 0;
      letter-spacing: 0.2px;
    }
    .sidebar-sub {
      font-family: 'Poppins', sans-serif;
      font-size: 13px;
      font-weight: 600;
      color: rgba(255,255,255,0.9);
      margin-bottom: 8px;
    }

    /* Make the actual widget text bigger where possible */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .css-1v0mbdj, 
    [data-testid="stSidebar"] .css-1d391kg {
      color: #fff !important;
    }

    /* Radio option text & selectbox text (best-effort selectors) */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stSelectbox div[role="listbox"] span,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stDateInput,
    [data-testid="stSidebar"] .stTextInput {
      font-size: 14px !important;
      font-weight: 700 !important;
      color: #fff !important;
    }

    /* Make the slider labels / ticks a bit bolder */
    [data-testid="stSidebar"] .css-1avcm0n, 
    [data-testid="stSidebar"] .css-1avcm0n * {
      color: #fff !important;
      font-weight: 700 !important;
    }

    /* Increase space between controls for clarity */
    [data-testid="stSidebar"] .stButton, 
    [data-testid="stSidebar"] .stRadio, 
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stSlider, 
    [data-testid="stSidebar"] .stDateInput, 
    [data-testid="stSidebar"] .stTextInput {
      margin-bottom: 14px !important;
    }

    /* Give the sidebar title a stronger look */
    .sidebar-title {
      font-family: 'Poppins', sans-serif;
      font-size: 20px;
      font-weight: 900;
      color: #ffffff;
      margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar header (HTML)
st.sidebar.markdown("<div class='sidebar-title'>Dashboard Controls</div>", unsafe_allow_html=True)

# Select Forecasting Method (label as HTML for larger white text)
st.sidebar.markdown("<div class='sidebar-label'>Select Forecasting Method</div>", unsafe_allow_html=True)
method = st.sidebar.selectbox(
    label="",  # label empty because we rendered it above
    options=["Prophet (Monthly)", "XGBoost (Weekly)"],
    key="method_select",
    format_func=lambda x: x
)

# Choose Medicine Category (render heading)
st.sidebar.markdown("<div class='sidebar-label'>Choose Medicine Category</div>", unsafe_allow_html=True)

# Use an explicit label above the radio (so the option group is visually separated)
st.sidebar.markdown("<div class='sidebar-sub'>Select one of the medicine categories below</div>", unsafe_allow_html=True)

# Render radio with empty label (we already showed the big title)
medicine = st.sidebar.radio(
    label="",
    options=list(MEDICINE_CATEGORIES.keys()),
    format_func=lambda x: MEDICINE_CATEGORIES.get(x, x),
    key="medicine_radio"
)

# Forecast settings
if method.startswith("Prophet"):
    monthly_df = load_monthly_data()
    if monthly_df.empty:
        st.sidebar.error(f"Monthly data file not found or empty: {PROPHET_DATA}")

    # Date fields removed as they are no longer required
    # Adding some spacing for better visual hierarchy
    st.sidebar.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<div class='sidebar-label'>Forecast Horizon (months)</div>", unsafe_allow_html=True)
    forecast_horizon = st.sidebar.slider(
        label="",
        min_value=1,
        max_value=36,
        value=12,
        key="forecast_horizon"
    )
else:
    weekly_df = load_weekly_data()
    if weekly_df.empty:
        st.sidebar.error(f"Weekly data file not found or empty: {XGB_DATA}")

    st.sidebar.markdown("<div class='sidebar-label'>Number of lag features (per series)</div>", unsafe_allow_html=True)
    lags = st.sidebar.slider("",
                             min_value=1, max_value=6, value=4, key="lags_slider")

    st.sidebar.markdown("<div class='sidebar-label'>Forecast horizon (weeks)</div>", unsafe_allow_html=True)
    forecast_horizon_weeks = st.sidebar.slider("",
                                              min_value=1, max_value=52, value=12, key="forecast_weeks")

    st.sidebar.markdown("<div class='sidebar-label'>Number of Forecast Iterations</div>", unsafe_allow_html=True)
    n_iters = st.sidebar.slider("",
                               min_value=1, max_value=5, value=3, key="n_iters")

# ---------------- Stock update toggle (styled red + black text) ----------------
# This CSS targets both the "Stock Update" sidebar button and the "Submit Update" form button,
# ensuring they are always red and their inner text is black and clearly visible.

st.markdown(
    f"""
    <style>
    /* Primary selector for the Stock Update sidebar button (aria-label equals the button label) */
    button[aria-label="Stock Update"] {{
        background: {THEME_BLUE} !important;
        color: #000000 !important;                     /* black text inside button */
        -webkit-text-fill-color: #000000 !important;   /* ensure on webkit browsers */
        font-weight: 800 !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: 0 6px 14px rgba(231,76,60,0.12) !important;
        width: 100% !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 15px !important;
        text-align: center;
        text-transform: none !important;
    }}

    /* Ensure hover/focus doesn't hide text or invert color */
    button[aria-label="Stock Update"]:focus {{
        color: #ffffff !important;
        -webkit-text-fill-color: #000000 !important;
        filter: none !important;
        transform: translateY(-1px);
    }}

    /* Form submit button inside sidebar form: "Submit Update" */
    button[aria-label="Submit Update"],
    button[aria-label="Submit Update"][type="submit"] {{
        background: {CANCEL_RED} !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 800 !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: 0 6px 14px rgba(231,76,60,0.12) !important;
        width: 100% !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 15px !important;
        text-align: center;
        text-transform: none !important;
    }}

    button[aria-label="Submit Update"]:hover,
    button[aria-label="Submit Update"]:focus {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        filter: none !important;
        transform: translateY(-1px);
    }}

    /* Force high contrast on any inner span elements */
    button[aria-label="Stock Update"] span,
    button[aria-label="Submit Update"] span {{
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }}

    /* In case Streamlit wraps the label differently, also target by button text */
    button:contains("Stock Update"),
    button:contains("Submit Update") {{
        color: #000000 !important;
    }}

    /* Safety: ensure other button rules (like download) are not overwritten unintentionally.
       We keep those buttons styled separately below. */
    </style>
    """,
    unsafe_allow_html=True,
)

# The actual toggle button (behavior unchanged)
if st.sidebar.button("Stock Update", key="stock_update_btn", use_container_width=True):
    st.session_state.show_stock_update = not st.session_state.show_stock_update


st.sidebar.markdown("---")
if "show_stock_update" not in st.session_state:
    st.session_state.show_stock_update = False
if "preview_payload" not in st.session_state:
    st.session_state.preview_payload = None
if "show_preview_modal" not in st.session_state:
    st.session_state.show_preview_modal = False

if st.session_state.show_stock_update:
    st.sidebar.subheader("Update Stock")
    with st.sidebar.form("update_stock_form_sidebar", clear_on_submit=False):
        new_tablet = st.text_input("Tablet description", placeholder="e.g. Paracetamol")
        new_volume = st.number_input("Volume to add", min_value=1, step=1, value=1)
        update_date = st.date_input("Date for update", value=datetime.today().date())
        submit_btn = st.form_submit_button("Submit Update")

    if submit_btn:
        if not new_tablet.strip():
            st.sidebar.error("Please provide a tablet description.")
        else:
            try:
                if method.startswith("Prophet"):
                    src_monthly = monthly_df if 'monthly_df' in locals() and not monthly_df.empty else pd.DataFrame(columns=['datum'])
                    action_m, chosen_m, df_month_preview = update_upcoming_file(UPCOMING_MONTHLY_COPY, src_monthly, new_tablet, new_volume, update_date)
                    st.sidebar.success(f"Monthly {action_m}: added {new_volume} to {chosen_m} on {update_date}.")

                    if "datum" in df_month_preview.columns:
                        df_month_preview["datum"] = pd.to_datetime(df_month_preview["datum"], errors="coerce")

                    payload = {
                        "df": df_month_preview,
                        "title": f"Monthly dataset preview: {os.path.basename(UPCOMING_MONTHLY_COPY)}",
                        "highlight": {"date": pd.to_datetime(update_date).strftime("%Y-%m-%d"), "col": chosen_m},
                        "file_bytes": df_month_preview.to_csv(index=False).encode("utf-8"),
                        "file_name": os.path.basename(UPCOMING_MONTHLY_COPY)
                    }
                    st.session_state.preview_payload = payload
                    st.session_state.show_preview_modal = True

                else:
                    src_weekly = weekly_df.reset_index() if 'weekly_df' in locals() and not weekly_df.empty else pd.DataFrame(columns=['datum'])
                    action_w, chosen_w, df_week_preview = update_upcoming_file(UPCOMING_WEEKLY_COPY, src_weekly, new_tablet, new_volume, update_date)
                    st.sidebar.success(f"Weekly {action_w}: added {new_volume} to {chosen_w} on {update_date}.")

                    if "datum" in df_week_preview.columns:
                        df_week_preview["datum"] = pd.to_datetime(df_week_preview["datum"], errors="coerce")

                    payload = {
                        "df": df_week_preview,
                        "title": f"Weekly dataset preview: {os.path.basename(UPCOMING_WEEKLY_COPY)}",
                        "highlight": {"date": pd.to_datetime(update_date).strftime("%Y-%m-%d"), "col": chosen_w},
                        "file_bytes": df_week_preview.to_csv(index=False).encode("utf-8"),
                        "file_name": os.path.basename(UPCOMING_WEEKLY_COPY)
                    }
                    st.session_state.preview_payload = payload
                    st.session_state.show_preview_modal = True

            except PermissionError:
                st.sidebar.error("Permission denied while saving files. Close Excel or retry.")
            except Exception as e:
                st.sidebar.error(f"Update failed: {e}")

st.markdown("---")

# ----------------- Prophet Flow -----------------
if method.startswith("Prophet"):
    st.subheader("Prophet-Monthly Forecast")
    if 'monthly_df' not in locals() or monthly_df.empty or 'datum' not in monthly_df.columns:
        st.error(f"Monthly data file missing or invalid: {PROPHET_DATA}")
    else:
        if medicine not in monthly_df.columns:
            st.warning(f"Selected medicine {medicine} not present in monthly data. Showing first available column instead.")
            available_cols = [c for c in monthly_df.columns if c != "datum"]
            if available_cols:
                medicine = available_cols[0]

        sample = monthly_df[['datum', medicine]].rename(columns={'datum': 'ds', medicine: 'y'})
        # Since start_date and end_date fields were removed, use all available data
        filtered_sample = sample.copy()

        model_path = os.path.join(PROPHET_MODEL_DIR, f"{medicine}_prophet.pkl")
        if not check_file_exists(model_path):
            st.error(f"Model file for {medicine} not found at {model_path}")
        else:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                model = None

            if model is not None:
                freq = 'M'
                future = model.make_future_dataframe(periods=forecast_horizon, freq=freq)
                forecast = model.predict(future)

                plot_x_actual, plot_y_actual = sample['ds'], sample['y']
                plot_x_forecast, plot_y_forecast = forecast['ds'], forecast['yhat']
                future_start = sample['ds'].max()
                future_forecast = forecast[forecast['ds'] > future_start]
                plot_x_future, plot_y_future = future_forecast['ds'], future_forecast['yhat']

                merged = pd.merge(filtered_sample, forecast, on='ds', how='inner')
                if merged.empty:
                    mape = rmse = r2 = accuracy = 0.0
                else:
                    actual = merged['y'].values
                    predicted = merged['yhat'].values
                    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)) if np.var(actual) != 0 else 0.0
                    accuracy = 100 - mape

                st.subheader(f"Model Performance for {medicine}")
                col1, col2, col3, col4 = st.columns(4, gap="small")
                with col1:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>Accuracy</div><div class='metric-value'>{accuracy:.2f}%</div></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>MAPE</div><div class='metric-value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>RMSE</div><div class='metric-value'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>R²</div><div class='metric-value'>{r2*100:.2f}%</div></div>", unsafe_allow_html=True)

                st.subheader("Actual vs Future Sales")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_x_actual, y=plot_y_actual, mode='lines+markers', name='Historical Actual', line=dict(color=THEME_BLUE)))
                fig.add_trace(go.Scatter(x=plot_x_forecast, y=plot_y_forecast, mode='lines+markers', name='Forecast', line=dict(color=THEME_PURPLE)))
                fig.add_trace(go.Scatter(x=plot_x_future, y=plot_y_future, mode='lines+markers', name='Future Forecast', line=dict(color='red', dash='dash')))
                fig.update_layout(title=f"Actual vs Future Sales for {medicine}", xaxis_title="Date", yaxis_title="Sales",
                                  template="plotly_white", font=dict(family="Poppins, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)

                # Gemini explanation for Prophet overall forecast trend
                try:
                    summary_series = future_forecast['yhat'] if not future_forecast.empty and 'yhat' in future_forecast.columns else forecast['yhat'].tail(max(1, forecast.shape[0]))
                    summary = build_trend_summary(summary_series)
                    explanation_para = get_forced_explanation(product_name=medicine, timeframe=f"{forecast_horizon}-month forecast (monthly)", summary=summary, n_sentences=4)
                    st.markdown("#### Forecast explanation")
                    st.markdown(explanation_para)
                except Exception:
                    local_para = build_local_explanation(product_name=medicine, timeframe=f"{forecast_horizon}-month forecast (monthly)", summary={"start":0,"end":0,"pct_change":0,"mean":0,"std":0,"n_points":0,"recent_slope":0})
                    st.markdown("#### Forecast explanation")
                    st.markdown(local_para)

                # Overall Forecast Trend (Prophet)
                try:
                    if not future_forecast.empty and 'yhat' in future_forecast.columns:
                        pf_series = future_forecast['yhat'].reset_index(drop=True)
                        trend_start = float(pf_series.iloc[0]) if len(pf_series) > 0 else 0.0
                        trend_end = float(pf_series.iloc[-1]) if len(pf_series) > 0 else 0.0
                        perc_change = ((trend_end - trend_start) / (trend_start if trend_start != 0 else 1)) * 100
                        st.subheader("Overall Forecast Trend (Prophet)")
                        if perc_change > 2:
                            st.success(f"Increasing trend ({perc_change:.2f}%).")
                        elif perc_change < -2:
                            st.error(f"Decreasing trend ({perc_change:.2f}%).")
                        else:
                            st.info(f"Overall, the forecast indicates a stable trend in sales ({perc_change:.2f}%).")
                    else:
                        st.subheader("Overall Forecast Trend (Prophet)")
                        st.info("No future forecast produced to compute trend.")
                except Exception:
                    st.warning("Could not compute overall trend for Prophet.")

                try:
                    fig_components = model.plot_components(forecast)
                    st.pyplot(fig_components)
                except Exception:
                    st.warning("Could not display prophet components.")

# ----------------- XGBoost Flow -----------------
else:
    st.subheader("XGBoost-Weekly Forecast")
    if 'weekly_df' not in locals() or weekly_df.empty:
        st.error(f"Weekly data file missing or invalid: {XGB_DATA}")
    else:
        categories = [c for c in weekly_df.columns if c in MEDICINE_CATEGORIES.keys()]
        if not categories:
            st.error("No medicine columns found in weekly data that match MEDICINE_CATEGORIES.")
        else:
            if medicine not in categories:
                st.warning(f"Selected medicine {medicine} not present in weekly data. Showing {categories[0]} instead.")
                medicine_weekly = categories[0]
            else:
                medicine_weekly = medicine

            df_prep = prepare_data_xgb(weekly_df, categories, lags=lags)
            if df_prep.empty:
                st.warning("Not enough historical rows after creating lag features. Reduce the number of lag features or add more data.")
            else:
                models = {}
                with st.spinner("Training models..."):
                    try:
                        for col in categories:
                            features = ['year', 'month', 'week'] + [f'{col}_lag{i}' for i in range(1, lags + 1)]
                            if not set(features).issubset(df_prep.columns):
                                continue
                            X = df_prep[features]
                            y = df_prep[col]
                            models[col] = train_xgb(X, y)
                    except Exception as e:
                        st.error(f"Model training failed: {e}")

                test_size = min(12, int(len(df_prep) * 0.2))
                if len(df_prep) > test_size:
                    train_df = df_prep.iloc[:-test_size]
                    test_df = df_prep.iloc[-test_size:]
                else:
                    train_df = df_prep
                    test_df = df_prep

                y_test = test_df[categories] if not test_df.empty else pd.DataFrame()
                y_pred = pd.DataFrame(index=test_df.index, columns=categories)
                for col in categories:
                    features = ['year', 'month', 'week'] + [f'{col}_lag{i}' for i in range(1, lags + 1)]
                    if col in models and not test_df.empty and set(features).issubset(test_df.columns):
                        X_test = test_df[features]
                        try:
                            y_pred[col] = models[col].predict(xgb.DMatrix(X_test))
                        except Exception:
                            y_pred[col] = np.nan

                col = medicine_weekly
                if not y_test.empty and col in y_pred:
                    actual = y_test[col].values
                    predicted = y_pred[col].astype(float).values
                    mask = ~np.isnan(predicted)
                    if mask.any():
                        actual_masked = actual[mask]
                        predicted_masked = predicted[mask]
                        mape = np.mean(np.abs((actual_masked - predicted_masked) / np.where(actual_masked == 0, 1, actual_masked))) * 100
                        rmse = np.sqrt(np.mean((actual_masked - predicted_masked) ** 2))
                        r2 = 1 - (np.sum((actual_masked - predicted_masked) ** 2) / np.sum((actual_masked - np.mean(actual_masked)) ** 2)) if np.var(actual_masked) != 0 else 0.0
                        accuracy = max(0.0, 100 - mape)
                    else:
                        mape = rmse = r2 = accuracy = 0.0
                else:
                    mape = rmse = r2 = accuracy = 0.0

                st.subheader(f"Model Performance for {col}")
                col1, col2, col3, col4 = st.columns(4, gap="small")
                with col1:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>Accuracy</div><div class='metric-value'>{accuracy:.2f}%</div></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>MAPE</div><div class='metric-value'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>RMSE</div><div class='metric-value'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"<div class='metric-box'><div class='metric-label'>R²</div><div class='metric-value'>{r2*100:.2f}%</div></div>", unsafe_allow_html=True)

                sim_data_df = weekly_df.copy()
                if not isinstance(sim_data_df.index, pd.DatetimeIndex):
                    sim_data_df.index = pd.to_datetime(sim_data_df.index)
                sim_data_df = sim_data_df.sort_index()
                sim_model = models.get(col)

                try:
                    display_start = sim_data_df.index.min() if sim_data_df.index.min() is not None else sim_data_df.index.min()
                except Exception:
                    display_start = sim_data_df.index.min()
                sim_display = sim_data_df[sim_data_df.index >= display_start]

                final_forecast = pd.Series(dtype=float)
                if sim_model is None:
                    st.warning(f"No trained model for {col} available.")
                else:
                    sim_local = sim_data_df.copy()
                    sim_local_model = sim_model
                    for i in range(n_iters):
                        forecast_series = forecast_col_xgb(col, forecast_horizon_weeks, sim_local_model, sim_local, lags=lags, freq='W')
                        final_forecast = forecast_series.copy()
                        forecast_df = pd.DataFrame({col: forecast_series})
                        sim_local = pd.concat([sim_local, forecast_df])
                        new_prep = prepare_data_xgb(sim_local, categories, lags=lags)
                        if len(new_prep) >= lags + 1 and col in new_prep.columns:
                            features = ['year', 'month', 'week'] + [f'{col}_lag{j}' for j in range(1, lags + 1)]
                            new_X = new_prep.tail(12)[features] if len(new_prep) >= 12 else new_prep[features]
                            new_y = new_prep.tail(12)[col] if len(new_prep) >= 12 else new_prep[col]
                            try:
                                sim_local_model = train_xgb(new_X, new_y, sim_local_model)
                            except Exception as e:
                                st.warning(f"Incremental retrain failed on iter {i+1}: {e}")

                if not final_forecast.empty and not sim_display.empty:
                    final_forecast = pd.Series(final_forecast.values, index=final_forecast.index)
                    last_hist = sim_display.index.max()
                    desired_start = last_hist + pd.Timedelta(weeks=1)
                    try:
                        new_index = pd.date_range(start=desired_start, periods=len(final_forecast), freq='W')
                        final_forecast.index = new_index
                    except Exception:
                        final_forecast.index = pd.to_datetime(final_forecast.index)
                    final_forecast = final_forecast.sort_index()
                elif not final_forecast.empty:
                    final_forecast.index = pd.to_datetime(final_forecast.index)

                st.subheader(f" Actual & Future Sales ")
                fig = go.Figure()
                if col in sim_display.columns:
                    fig.add_trace(go.Scatter(x=sim_display.index, y=sim_display[col], mode='lines+markers', name='Historical Actual', line=dict(color=THEME_BLUE)))
                if col in y_pred and y_pred[col].notna().any():
                    try:
                        y_pred.index = pd.to_datetime(y_pred.index)
                    except Exception:
                        pass
                    fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred[col], mode='lines+markers', name='Predicted (test)', line=dict(color=THEME_PURPLE)))
                if not final_forecast.empty:
                    # Connect the future forecast to the last historical point
                    last_x = sim_display.index[-1]
                    last_y = sim_display[col].iloc[-1]
                    future_x = [last_x] + list(final_forecast.index)
                    future_y = [last_y] + list(final_forecast.values)
                    fig.add_trace(go.Scatter(x=future_x, y=future_y, mode='lines+markers', name='Future Forecast', line=dict(color='red', dash='dash')))
                    x_min = sim_display.index.min() if not sim_display.empty else None
                    x_max = final_forecast.index.max()
                    if x_min is not None:
                        fig.update_xaxes(range=[x_min, x_max + pd.Timedelta(days=7)])
                fig.update_layout(title=f"Actual vs Future Sales for {col}", xaxis_title="Date", yaxis_title="Sales",
                                  template="plotly_white", font=dict(family="Poppins, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)

                # Gemini explanation for XGBoost overall forecast trend
                try:
                    if not final_forecast.empty:
                        summary = build_trend_summary(final_forecast)
                        explanation_para = get_forced_explanation(product_name=col, timeframe=f"{forecast_horizon_weeks}-week forecast (weekly)", summary=summary, n_sentences=4)
                        st.markdown("#### Forecast explanation")
                        st.markdown(explanation_para)
                    else:
                        local_para = build_local_explanation(product_name=col, timeframe=f"{forecast_horizon_weeks}-week forecast (weekly)", summary={"start":0,"end":0,"pct_change":0,"mean":0,"std":0,"n_points":0,"recent_slope":0})
                        st.markdown("#### Forecast explanation")
                        st.markdown(local_para)
                except Exception:
                    pass

                if not final_forecast.empty:
                    trend_start = final_forecast.iloc[0]
                    trend_end = final_forecast.iloc[-1]
                    perc_change = ((trend_end - trend_start) / (trend_start if trend_start != 0 else 1)) * 100
                    st.subheader("Overall Forecast Trend (XGBoost)")
                    if perc_change > 2:
                        st.success(f"Increasing trend ({perc_change:.2f}%).")
                    elif perc_change < -2:
                        st.error(f"Decreasing trend ({perc_change:.2f}%).")
                    else:
                        st.info(f"Overall, the forecast indicates a stable trend in sales ({perc_change:.2f}%).")
                else:
                    st.info("No future forecast produced to compute trend.")

st.markdown("---")

# ----------------- Render preview overlay at end (if requested) -----------------
preview_payload = st.session_state.get("preview_payload")
preview_visible = bool(st.session_state.get("show_preview_modal") and preview_payload)

preview_container = st.container()
download_container = st.container()

# Additional CSS for other buttons (kept separate)
st.markdown(
    """
    <style>
    .stDownloadButton>button { background: #87CEEB !important; color: #000000 !important; border-radius: 8px !important; padding: 8px 14px !important; font-weight:700 !important; font-family:'Poppins',sans-serif !important; }
    .stButton>button { border-radius:8px !important; padding:8px 12px !important; font-weight:700 !important; font-family:'Poppins',sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if preview_visible and preview_payload:
    with preview_container:
        df_preview = preview_payload.get("df")
        highlight = preview_payload.get("highlight", None)
        title = preview_payload.get("title", "Preview")

        try:
            styled = style_preview_dataframe(df_preview, highlight=highlight)
            html = styled.to_html() if hasattr(styled, "to_html") else styled.render()
            st.markdown(f"### {title}")
            # The styled HTML already contains table; inject into container with wrapper so our CSS applies
            st.markdown(f'<div class="dataframe-preview" style="font-family: Poppins, sans-serif;">{html}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Styled preview failed: {e}. Falling back to plain preview.")
            html = build_preview_html(df_preview, title=title, highlight=highlight, max_rows=2000)
            components.html(html, height=700, scrolling=True)

        st.download_button(
            label="Download (CSV)",
            data=preview_payload["file_bytes"],
            file_name=preview_payload["file_name"],
            mime="text/csv",
            key="preview_direct_download"
        )

else:
    if preview_payload:
        with download_container:
            st.download_button(
                label="Download (CSV)",
                data=preview_payload["file_bytes"],
                file_name=preview_payload["file_name"],
                mime="text/csv",
                key="preview_direct_download_alone"
            )

# End of script