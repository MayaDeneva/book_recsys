"""Render results tables and splice them into the model report."""
import pandas as pd


def results_to_markdown(df: pd.DataFrame) -> str:
    """Render a results frame (index = method, columns = metrics) as a markdown table."""
    cols = list(df.columns)
    lines = [
        "| method | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for method, row in df.iterrows():
        cells = " | ".join(f"{row[c]:.4f}" for c in cols)
        lines.append(f"| {method} | {cells} |")
    return "\n".join(lines)


def splice_section(text: str, start: str, end: str, content: str) -> str:
    """Replace the text between the start and end markers (markers kept) with content."""
    s = text.index(start) + len(start)
    e = text.index(end)
    return text[:s] + "\n" + content + "\n" + text[e:]
