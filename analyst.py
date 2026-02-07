#Step Data Loading
import io
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import textwrap
import sys
import os

# Optional duckdb import (not required)
try:
    import duckdb
except Exception:
    duckdb = None

# ----------------- Load data -----------------
def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample

def load_data(file_or_path) -> pd.DataFrame:
    """
    Accepts Streamlit UploadedFile, path string/Path, or file-like object.
    Returns pandas DataFrame.
    """
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        s = p.suffix.lower()
        if s == ".csv":
            return pd.read_csv(p)
        if s in {".xls", ".xlsx"}:
            return pd.read_excel(p)
        if s == ".json":
            return pd.read_json(p)
        return pd.read_csv(p)

    # file-like (UploadedFile)
    name = getattr(file_or_path, "name", None)
    suffix = Path(name).suffix.lower() if name else None
    raw = file_or_path.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    bio = io.BytesIO(raw)

    if suffix == ".csv" or (suffix is None and _looks_like_csv(raw)):
        bio.seek(0); return pd.read_csv(bio)
    if suffix in {".xls", ".xlsx"}:
        bio.seek(0); return pd.read_excel(bio)
    if suffix == ".json":
        bio.seek(0); return pd.read_json(bio)
    # fallback
    bio.seek(0)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0); return pd.read_json(bio)

#Step 2: Deterministic Intelligence (Prompt Suggestions)
def _detect_column_types(df: pd.DataFrame):
   numeric = df.select_dtypes(include=[np.number]).columns.tolist()
   datetime = []
   # try to infer datetime columns
   for c in df.columns:
       if np.issubdtype(df[c].dtype, np.datetime64):
           datetime.append(c)
       else:
           # try to parse small sample as date
           try:
               sample = df[c].dropna().astype(str).iloc[:20]
               parsed = pd.to_datetime(sample, errors="coerce")
               if parsed.notna().sum() >= max(1, min(5, len(sample)//2)):
                   datetime.append(c)
           except Exception:
               pass
   # categoricals: low cardinality non-numeric
   categorical = [c for c in df.columns if c not in numeric + datetime and df[c].nunique(dropna=True) <= 50]
   return {"numeric": numeric, "datetime": datetime, "categorical": categorical}

def suggest_prompts(df: pd.DataFrame, max_suggestions: int = 8):
    """
    Return a list of helpful, ready-to-run prompt strings for the dataset.
    Deterministic and works without any LLM.
    """
    types = _detect_column_types(df)
    numeric = types["numeric"]
    datetime = types["datetime"]
    categorical = types["categorical"]

    suggestions = []
    # Basic summary
    suggestions.append("Summarize the dataset in 5 bullet points (rows, columns, missing values, numeric columns, top categorical).")
    # Top value queries
    if categorical:
        col = categorical[0]
        suggestions.append(f"Show the top 10 counts for the categorical column '{col}'.")
    # Numeric summaries
    if numeric:
        suggestions.append(f"Show summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.")
        col = numeric[0]
        suggestions.append(f"Create a histogram of the numeric column '{col}'.")
        if len(numeric) >= 2:
            suggestions.append(f"Create a scatter plot comparing '{numeric[0]}' (x) vs '{numeric[1]}' (y).")
        suggestions.append(f"Show the top 10 rows sorted by '{col}' descending.")
    # Time series
    if datetime:
        dcol = datetime[0]
        # choose a numeric for aggregation if exists
        ag = numeric[0] if numeric else None
        if ag:
            suggestions.append(f"Create a time series of monthly sum of '{ag}' using the datetime column '{dcol}'.")
        else:
            suggestions.append(f"Show counts per month using the datetime column '{dcol}'.")
    # Correlation
    if len(numeric) >= 2:
        suggestions.append("Show the correlation matrix heatmap for numeric columns.")
    # Generic top-k
    suggestions.append("Find rows that look like anomalies using z-score > 3 on numeric columns and show top 20.")
    # limit suggestions
    return suggestions[:max_suggestions]

#Step 3: The Translator functions
def prompt_to_code(prompt: str, df: pd.DataFrame):
    """
    Convert known prompt templates into runnable python code strings.
    If the prompt is custom/unrecognized, return None (so UI can send to LLM instead).
    """
    p = prompt.strip().lower()

    # Summary
    if p.startswith("summarize the dataset"):
        code = textwrap.dedent("""
            # produce a short summary as printed text
            info = []
            info.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            info.append("Column types: " + ", ".join([f"{c}:{str(df[c].dtype)[:10]}" for c in df.columns[:10]]))
            miss = df.isnull().sum().sort_values(ascending=False).head(10)
            info.append("Top missing: " + ", ".join([f"{idx}:{val}" for idx,val in miss.items() if val>0]))
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            info.append(f"Numeric columns count: {len(numeric)}")
            # print concise bullets
            result = "\\n".join(["- "+i for i in info])
        """)
        return code

    # Top counts for categorical
    if "top 10 counts for the categorical column" in p or "top 10 counts" in p and "'" in p:
        # try to extract column name between quotes
        import re
        m = re.search(r"'([^']+)'", prompt)
        if not m:
            m = re.search(r'"([^"]+)"', prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                # top 10 counts for '{col}'
                result = df['{col}'].value_counts(dropna=False).head(10).reset_index()
                result.columns = ['value','count']
            """)
            return code

    # Summary statistics for numeric
    if "summary statistics" in p or "describe" in p:
        code = textwrap.dedent("""
            result = df.select_dtypes(include=['number']).describe().T
        """)
        return code

    # Histogram
    if p.startswith("create a histogram of the numeric column") or "histogram of the numeric column" in p:
        import re
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                # histogram for '{col}'
                fig = px.histogram(df, x='{col}', nbins=30, title='Histogram of {col}')
                result = fig
            """)
            return code

    # Scatter plot
    if "scatter plot comparing" in p and "vs" in p:
        import re
        m = re.search(r"'([^']+)' \(x\) vs '([^']+)' \(y\)", prompt)
        if m:
            xcol, ycol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                fig = px.scatter(df, x='{xcol}', y='{ycol}', title='{ycol} vs {xcol}')
                result = fig
            """)
            return code

    # Top N rows sorted by col
    if p.startswith("show the top 10 rows sorted by"):
        import re
        m = re.search(r"by '([^']+)'", prompt)
        if m:
            col = m.group(1)
            code = textwrap.dedent(f"""
                result = df.sort_values('{col}', ascending=False).head(10).reset_index(drop=True)
            """)
            return code

    # Time series monthly sum
    if "monthly sum" in p and "using the datetime column" in p:
        import re
        m = re.search(r"sum of '([^']+)' using the datetime column '([^']+)'", prompt)
        if m:
            ag, dcol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}'])
                res = res.set_index('{dcol}').resample('M')['{ag}'].sum().reset_index()
                fig = px.line(res, x='{dcol}', y='{ag}', title='Monthly sum of {ag}')
                result = fig
            """)
            return code

    # Counts per month (datetime only)
    if "counts per month using the datetime column" in p:
        import re
        m = re.search(r"datetime column '([^']+)'", prompt)
        dcol = m.group(1) if m else None
        if dcol:
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}']).set_index('{dcol}').resample('M').size().reset_index(name='count')
                fig = px.line(res, x='{dcol}', y='count', title='Counts per month')
                result = fig
            """)
            return code

    # Correlation heatmap
    if "correlation matrix heatmap" in p or "correlation heatmap" in p:
        code = textwrap.dedent("""
            corr = df.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation matrix')
            result = fig
        """)
        return code

    # Anomaly detection using z-score
    if "anomalies" in p and "z-score" in p:
        code = textwrap.dedent("""
            from scipy import stats
            num = df.select_dtypes(include=['number']).dropna()
            if num.shape[1]==0:
                result = pd.DataFrame()
            else:
                z = np.abs(stats.zscore(num.select_dtypes(include=['number'])))
                mask = (z > 3).any(axis=1)
                result = df.loc[mask].head(20).reset_index(drop=True)
        """)
        return code

    # Unknown / custom prompts -> return None
    return None

#The Execution Engine 
def run_code(df: pd.DataFrame, code: str):
    """
    Execute code string in a restricted local namespace.
    Returns a dict:
      - {"type":"text","output":...}
      - {"type":"dataframe","df": pandas.DataFrame}
      - {"type":"plotly","fig": plotly.graph_objects.Figure}
      - {"type":"image","path": path_to_png} (fallback)
    """
    # prepare namespace
    local_ns = {"pd": pd, "np": np, "df": df, "plt": plt, "px": px}
    # capture prints
    old_stdout = sys.stdout
    stdout_buf = io.StringIO()
    sys.stdout = stdout_buf
    try:
        # run
        # Pass local_ns as globals so that functions/comprehensions can access variables
        exec(code, local_ns)
        
        # Check for plotly figure (in result variable or generic 'fig' variable)
        if "result" in local_ns:
            res = local_ns["result"]
            # If result is a Plotly figure
            if hasattr(res, "show") and hasattr(res, "to_json"): 
                return {"type": "plotly", "fig": res}
            # If result is a DataFrame
            if isinstance(res, pd.DataFrame):
                return {"type": "dataframe", "df": res}
            # If result is a string/other
            if not isinstance(res, (pd.DataFrame, type(plt))):
                # Don't return plot object as text, but return text if it's string
                 if isinstance(res, str):
                    return {"type": "text", "output": res}

        # Check for 'fig' variable explicitly if 'result' wasn't useful
        if "fig" in local_ns:
             res = local_ns["fig"]
             if hasattr(res, "show") and hasattr(res, "to_json"):
                 return {"type": "plotly", "fig": res}

        # Fallback: check for matplotlib figure
        figs = plt.get_fignums()
        if figs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name, bbox_inches="tight", dpi=150)
                plt.close("all")
                return {"type": "image", "path": f.name}
        
        # If result was present but we didn't return yet (e.g. non-string non-df)
        if "result" in local_ns:
             return {"type": "text", "output": str(local_ns["result"])}

        # otherwise, return captured stdout
        out = stdout_buf.getvalue().strip()
        if out:
            return {"type": "text", "output": out}
        
        return {"type": "text", "output": "Execution finished. No result produced."}
    except Exception as e:
        return {"type": "text", "output": f"Execution error: {e}"}
    finally:
        sys.stdout = old_stdout
    
#Step : The LLM Connector 
def ask_llm(prompt: str, model: str = "llama3.1", timeout: int = 60) -> str:
    """
    Send prompt to local Ollama via CLI. Returns stdout text.
    If ollama is not installed or fails, returns an error string starting with [LLM...].
    Expect the model to return code inside ```python blocks.
    """
    try:
        proc = subprocess.run(["ollama", "run", model], input=prompt.encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        if not out and err:
            return f"[LLM-error] {err}"
        return out
    except FileNotFoundError:
        return "[LLM-missing] ollama not found on PATH."
    except Exception as e:
        return f"[LLM-failed] {e}"