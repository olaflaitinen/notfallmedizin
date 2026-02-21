# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson LundstrÃ¶m-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark report generation.

Produces structured plain-text or Markdown reports summarising model
performance, comparisons, and dataset characteristics.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportSection:
    """A single section of a benchmark report.

    Attributes
    ----------
    title : str
    content : str
    tables : list of dict
    """

    title: str = ""
    content: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)


class BenchmarkReport:
    """Structured benchmark report builder.

    Collects sections and renders them as Markdown text.
    """

    def __init__(self, title: str = "Benchmark Report") -> None:
        self.title = title
        self.sections: List[ReportSection] = []
        self._metadata: Dict[str, str] = {
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "library": "notfallmedizin",
        }

    def add_section(
        self,
        title: str,
        content: str = "",
        tables: Optional[List[Dict[str, Any]]] = None,
    ) -> "BenchmarkReport":
        """Append a section to the report.

        Parameters
        ----------
        title : str
        content : str
        tables : list of dict, optional

        Returns
        -------
        self
        """
        self.sections.append(
            ReportSection(title=title, content=content, tables=tables or [])
        )
        return self

    def add_dataset_summary(
        self,
        name: str,
        n_samples: int,
        n_features: int,
        class_distribution: Optional[Dict[str, int]] = None,
    ) -> "BenchmarkReport":
        """Add a dataset summary section.

        Parameters
        ----------
        name : str
        n_samples : int
        n_features : int
        class_distribution : dict, optional

        Returns
        -------
        self
        """
        lines = [
            f"Dataset: {name}",
            f"Samples: {n_samples}",
            f"Features: {n_features}",
        ]
        if class_distribution:
            lines.append("Class distribution:")
            for cls, cnt in class_distribution.items():
                lines.append(f"  {cls}: {cnt} ({cnt / n_samples * 100:.1f}%)")

        self.add_section("Dataset Summary", "\n".join(lines))
        return self

    def add_model_results(
        self,
        model_name: str,
        metrics: Dict[str, float],
    ) -> "BenchmarkReport":
        """Add model evaluation results.

        Parameters
        ----------
        model_name : str
        metrics : dict

        Returns
        -------
        self
        """
        table = [{"metric": k, "value": f"{v:.6f}"} for k, v in metrics.items()]
        self.add_section(f"Results: {model_name}", tables=[table])
        return self

    def render_markdown(self) -> str:
        """Render the report as Markdown text.

        Returns
        -------
        str
        """
        lines: List[str] = [f"# {self.title}", ""]

        for key, val in self._metadata.items():
            lines.append(f"**{key}**: {val}  ")
        lines.append("")

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            if section.content:
                lines.append(section.content)
                lines.append("")
            for table in section.tables:
                if isinstance(table, list) and len(table) > 0:
                    headers = list(table[0].keys())
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join("---" for _ in headers) + " |")
                    for row in table:
                        vals = [str(row.get(h, "")) for h in headers]
                        lines.append("| " + " | ".join(vals) + " |")
                    lines.append("")

        return "\n".join(lines)

    def render_text(self) -> str:
        """Render the report as plain text.

        Returns
        -------
        str
        """
        lines: List[str] = [self.title, "=" * len(self.title), ""]

        for key, val in self._metadata.items():
            lines.append(f"{key}: {val}")
        lines.append("")

        for section in self.sections:
            lines.append(section.title)
            lines.append("-" * len(section.title))
            if section.content:
                lines.append(section.content)
            for table in section.tables:
                if isinstance(table, list) and len(table) > 0:
                    headers = list(table[0].keys())
                    widths = [
                        max(len(h), max(len(str(r.get(h, ""))) for r in table))
                        for h in headers
                    ]
                    header_line = "  ".join(
                        h.ljust(w) for h, w in zip(headers, widths)
                    )
                    lines.append(header_line)
                    lines.append("  ".join("-" * w for w in widths))
                    for row in table:
                        row_line = "  ".join(
                            str(row.get(h, "")).ljust(w)
                            for h, w in zip(headers, widths)
                        )
                        lines.append(row_line)
            lines.append("")

        return "\n".join(lines)
