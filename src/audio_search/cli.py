from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table


app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()

DEFAULT_BASE = "http://localhost:8000"


def _client(base: str) -> httpx.Client:
    return httpx.Client(base_url=base, timeout=600.0)


@app.command()
def health(base: str = DEFAULT_BASE):
    """Hit /health and print the service status."""
    with _client(base) as c:
        r = c.get("/health")
        r.raise_for_status()
        console.print_json(json.dumps(r.json()))


@app.command()
def ingest(
    dataset: str = typer.Option(..., "--dataset", "-d", help="commonvoice | librispeech"),
    limit: int = typer.Option(500, "--limit", "-n"),
    batch_size: int = typer.Option(32, "--batch-size", "-b"),
    resume: bool = typer.Option(False, "--resume"),
    skip_audio: bool = typer.Option(False, "--skip-audio", help="text-only ingest (fast smoke)"),
    base: str = DEFAULT_BASE,
):
    """Run ingestion against the service."""
    payload = {"dataset": dataset, "limit": limit, "batch_size": batch_size, "resume": resume, "skip_audio": skip_audio}
    with _client(base) as c:
        r = c.post("/ingest", json=payload)
        r.raise_for_status()
        console.print_json(json.dumps(r.json()))


@app.command()
def search(
    q: str = typer.Argument(..., help="query string"),
    k: int = typer.Option(10, "--k", "-k"),
    sources: str = typer.Option("text,audio,bm25", "--sources", "-s"),
    source_filter: Optional[str] = typer.Option(None, "--source-filter"),
    base: str = DEFAULT_BASE,
):
    """Run a text search against the service."""
    params = {"q": q, "k": k, "sources": sources}
    if source_filter:
        params["source_filter"] = source_filter
    with _client(base) as c:
        r = c.get("/search", params=params)
        r.raise_for_status()
        data = r.json()
    _render_hits(data)


@app.command("search-by-audio")
def search_by_audio(
    audio_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    k: int = typer.Option(10, "--k", "-k"),
    source_filter: Optional[str] = typer.Option(None, "--source-filter"),
    base: str = DEFAULT_BASE,
):
    """Audio-to-audio search."""
    payload = {"audio_path": str(audio_path.resolve()), "k": k, "source_filter": source_filter}
    with _client(base) as c:
        r = c.post("/search-by-audio", json=payload)
        r.raise_for_status()
        data = r.json()
    _render_hits_audio(data)


@app.command("list")
def list_cmd(
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=200, help="page size"),
    cursor: Optional[str] = typer.Option(None, "--cursor", "-c", help="opaque cursor from previous page"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="filter: librispeech | fleurs | ..."),
    auto: bool = typer.Option(False, "--auto", help="auto-paginate through all pages (press enter between pages)"),
    base: str = DEFAULT_BASE,
):
    """Browse indexed clips with cursor-based pagination.

    Examples:
      audio-search list --limit 10                    # first page
      audio-search list --limit 10 --cursor ls_xxx    # next page
      audio-search list --auto                        # interactive walk
    """
    with _client(base) as c:
        next_cursor = cursor
        page_no = 0
        while True:
            page_no += 1
            params: dict = {"limit": limit}
            if next_cursor:
                params["cursor"] = next_cursor
            if source:
                params["source"] = source
            r = c.get("/clips", params=params)
            r.raise_for_status()
            data = r.json()
            _render_clips_page(data, page_no)
            next_cursor = data.get("next_cursor")
            if not next_cursor:
                console.print("[dim]— end of listing —[/dim]")
                break
            if not auto:
                console.print(f"[dim]next cursor:[/dim] [bold]{next_cursor}[/bold]")
                break
            try:
                input("press enter for next page (Ctrl+C to stop)…  ")
            except (KeyboardInterrupt, EOFError):
                console.print()
                break


def _render_clips_page(data: dict, page_no: int) -> None:
    n = len(data["clips"])
    total = data.get("total", "?")
    console.print(f"[dim]page[/dim] {page_no}   [dim]rows[/dim] {n}/{total}   "
                  f"[dim]cursor[/dim] {'-' if not data.get('next_cursor') else data['next_cursor']}")
    tbl = Table(show_lines=False)
    tbl.add_column("id")
    tbl.add_column("src")
    tbl.add_column("spk")
    tbl.add_column("dur", justify="right")
    tbl.add_column("transcript", overflow="fold")
    for c in data["clips"]:
        tbl.add_row(
            c["id"],
            c.get("source", ""),
            c.get("speaker_id", "")[:10],
            f"{c.get('duration_s', 0.0):.1f}s",
            c.get("transcript", ""),
        )
    console.print(tbl)


@app.command()
def eval(
    n_auto: int = typer.Option(200, "--n", help="number of auto-generated probes"),
    seed: int = typer.Option(42, "--seed"),
    base: str = DEFAULT_BASE,
):
    """Run the full eval harness (auto + hand probes × baseline / +audio / +bm25)."""
    with _client(base) as c:
        r = c.post("/eval", json={"n_auto": n_auto, "seed": seed})
        r.raise_for_status()
        _render_eval(r.json())


def _render_eval(summary: dict) -> None:
    console.print(f"[dim]probes:[/dim] {summary['n_probes_total']} "
                  f"([dim]auto={summary['n_auto']}, hand={summary['n_hand']}[/dim])")
    tbl = Table(title="Ablation — overall")
    tbl.add_column("config")
    tbl.add_column("R@1", justify="right")
    tbl.add_column("R@5", justify="right")
    tbl.add_column("R@10", justify="right")
    tbl.add_column("MRR", justify="right")
    for cfg, agg in summary["configs"].items():
        o = agg["overall"]
        tbl.add_row(cfg, f"{o['recall@1']:.3f}", f"{o['recall@5']:.3f}",
                    f"{o['recall@10']:.3f}", f"{o['mrr']:.3f}")
    console.print(tbl)

    sources = sorted({s for cfg in summary["configs"].values() for s in cfg["by_source"]})
    for src in sources:
        sub = Table(title=f"by source — {src}")
        sub.add_column("config")
        sub.add_column("n", justify="right")
        sub.add_column("R@1", justify="right")
        sub.add_column("R@5", justify="right")
        sub.add_column("R@10", justify="right")
        sub.add_column("MRR", justify="right")
        for cfg, agg in summary["configs"].items():
            a = agg["by_source"].get(src, {"n": 0, "recall@1": 0, "recall@5": 0, "recall@10": 0, "mrr": 0})
            sub.add_row(cfg, str(a["n"]), f"{a['recall@1']:.3f}",
                        f"{a['recall@5']:.3f}", f"{a['recall@10']:.3f}", f"{a['mrr']:.3f}")
        console.print(sub)


def _render_hits(data: dict) -> None:
    console.print(f"[dim]query:[/dim] [bold]{data['query']}[/bold]   "
                  f"[dim]timing:[/dim] {data['timing_ms']}")
    tbl = Table(show_lines=False)
    tbl.add_column("#", justify="right")
    tbl.add_column("rrf", justify="right")
    tbl.add_column("t", justify="right")
    tbl.add_column("a", justify="right")
    tbl.add_column("b", justify="right")
    tbl.add_column("src")
    tbl.add_column("id")
    tbl.add_column("transcript", overflow="fold")
    for i, h in enumerate(data["hits"]):
        tbl.add_row(
            str(i + 1),
            f"{h['rrf_score']:.4f}",
            "-" if h["text_rank"] is None else str(h["text_rank"]),
            "-" if h["audio_rank"] is None else str(h["audio_rank"]),
            "-" if h["bm25_rank"] is None else str(h["bm25_rank"]),
            h["source"],
            h["id"],
            h["transcript"],
        )
    console.print(tbl)


def _render_hits_audio(data: dict) -> None:
    console.print(f"[dim]audio query:[/dim] [bold]{data['query_audio_path']}[/bold]   "
                  f"[dim]timing:[/dim] {data['timing_ms']}")
    tbl = Table(show_lines=False)
    tbl.add_column("#", justify="right")
    tbl.add_column("audio_rank", justify="right")
    tbl.add_column("src")
    tbl.add_column("id")
    tbl.add_column("transcript", overflow="fold")
    for i, h in enumerate(data["hits"]):
        tbl.add_row(str(i + 1), str(h["audio_rank"]), h["source"], h["id"], h["transcript"])
    console.print(tbl)


if __name__ == "__main__":
    app()
