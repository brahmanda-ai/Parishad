"""Trace management for Parishad runs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

from ..roles.base import Trace


class TraceManager:
    """
    Manages execution traces for analysis and debugging.
    """
    
    def __init__(self, trace_dir: str | Path):
        """
        Initialize TraceManager.
        
        Args:
            trace_dir: Directory containing trace files
        """
        self.trace_dir = Path(trace_dir)
    
    def list_traces(self) -> list[str]:
        """List all trace IDs in the directory."""
        traces = []
        for path in self.trace_dir.glob("trace_*.json"):
            trace_id = path.stem.replace("trace_", "")
            traces.append(trace_id)
        return sorted(traces)
    
    def load_trace(self, trace_id: str) -> dict:
        """Load a trace by ID."""
        path = self.trace_dir / f"trace_{trace_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Trace not found: {trace_id}")
        
        with open(path) as f:
            return json.load(f)
    
    def iter_traces(self) -> Iterator[dict]:
        """Iterate over all traces."""
        for trace_id in self.list_traces():
            yield self.load_trace(trace_id)
    
    def get_summary(self) -> dict:
        """Get summary statistics for all traces."""
        traces = list(self.iter_traces())
        
        if not traces:
            return {"count": 0}
        
        total_tokens = sum(t.get("total_tokens", 0) for t in traces)
        total_latency = sum(t.get("total_latency_ms", 0) for t in traces)
        success_count = sum(1 for t in traces if t.get("success", False))
        
        return {
            "count": len(traces),
            "success_rate": success_count / len(traces),
            "avg_tokens": total_tokens / len(traces),
            "avg_latency_ms": total_latency / len(traces),
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
        }
    
    def filter_by_config(self, config: str) -> list[dict]:
        """Filter traces by configuration."""
        return [t for t in self.iter_traces() if t.get("config") == config]
    
    def filter_by_date(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[dict]:
        """Filter traces by date range."""
        results = []
        for trace in self.iter_traces():
            timestamp_str = trace.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                continue
            
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
            
            results.append(trace)
        
        return results
    
    def export_metrics(self, output_path: str | Path) -> None:
        """
        Export trace metrics to a CSV file.
        
        Useful for analysis and plotting.
        """
        import csv
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "query_id",
                "config",
                "timestamp",
                "total_tokens",
                "total_latency_ms",
                "budget_remaining",
                "retries",
                "success",
                "num_roles",
                "final_confidence"
            ])
            
            # Data
            for trace in self.iter_traces():
                final_answer = trace.get("final_answer", {})
                writer.writerow([
                    trace.get("query_id", ""),
                    trace.get("config", ""),
                    trace.get("timestamp", ""),
                    trace.get("total_tokens", 0),
                    trace.get("total_latency_ms", 0),
                    trace.get("budget_remaining", 0),
                    trace.get("retries", 0),
                    trace.get("success", False),
                    len(trace.get("roles", [])),
                    final_answer.get("confidence", 0)
                ])
