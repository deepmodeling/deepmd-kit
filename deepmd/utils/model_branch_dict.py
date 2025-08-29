# SPDX-License-Identifier: LGPL-3.0-or-later
import textwrap
from collections import (
    OrderedDict,
)
from typing import (
    Any,
    Optional,
)


def get_model_dict(model_dict: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Get the model branch alias dictionary from the model_dict.

    Returns
    -------
    model_alias_dict: dict
        A dictionary where the keys are the aliases and the values are the original model branch names.
    model_branch_dict: dict
        A dictionary where the keys are the original model branch names, and the values are dictionaries with:
        - alias
            the list of aliases of this model branch.
        - info
            the info dictionary of this model branch.
    """
    model_alias_dict = {}
    model_branch_dict = {}
    for key in model_dict:
        model_branch_dict[key] = {}
        model_alias_dict[key] = key
        alias_list = model_dict[key].get("model_branch_alias", [])
        model_branch_dict[key]["alias"] = alias_list
        branch_info = model_dict[key].get("info", {})
        model_branch_dict[key]["info"] = branch_info
        for alias in alias_list:
            assert alias not in model_alias_dict, (
                f"Alias {alias} for model_branch {key} already exists in model_branch {model_alias_dict[alias]}!"
            )
            model_alias_dict[alias] = key

    return model_alias_dict, model_branch_dict


# generated with GPT for formatted print
class OrderedDictTableWrapper:
    """
    A wrapper for pretty-printing an OrderedDict that has a specific structure.

    Expected structure:
        OrderedDict({
            "BranchName1": {
                "alias": ["A", "B"],                  # Required key: alias (list of strings)
                "info": {                             # Optional key: info (dict of arbitrary key-value pairs)
                    "description": "Some text",
                    "description2": "Some long text..."
                }
            },
            "BranchName2": {
                "alias": ["C"],
                "info": { "owner": "Alice" }
            },
            ...
        })

    Features:
    - Prints the data as an ASCII table with borders and aligned columns.
    - The first two columns are fixed: "Model Branch Name" and "Alias".
    - The remaining columns are all unique keys found in `info` across all branches (order preserved by first occurrence).
    - Long text in cells is automatically wrapped to fit the column width, except column 1 & 2 auto-expanding to the **maximum content length** in that column.
    - Missing info values are shown as empty strings.
    """

    def __init__(
        self, data: "OrderedDict[str, dict[str, Any]]", col_width: int = 30
    ) -> None:
        """
        Initialize the table wrapper.

        Args:
            data: OrderedDict containing the branch data.
            col_width: Maximum width of each column (characters). Longer text will wrap.
        """
        # Ensure we are working with an OrderedDict to preserve branch order
        if not isinstance(data, OrderedDict):
            data = OrderedDict(data)
        self.data = data
        self.col_width = col_width

        # Collect all unique keys from "info" across all branches in order of first appearance
        seen = set()
        self.info_keys: list[str] = []
        for _, payload in self.data.items():
            info = payload.get("info") or {}
            for k in info.keys():
                if k not in seen:
                    seen.add(k)
                    self.info_keys.append(k)

        # Construct table header: fixed columns + dynamic info keys
        self.headers: list[str] = ["Model Branch", "Alias", *self.info_keys]

    def _wrap_cell(self, text: Any, width: Optional[int] = None) -> list[str]:
        """
        Convert a cell value into a list of wrapped text lines.

        Args:
            text: Any value that will be converted to a string.
            width: Optional custom wrap width. If None, defaults to `self.col_width`.

        Returns
        -------
        A list of strings, each representing one wrapped line of the cell.
        """
        text = "" if text is None else str(text)
        eff_width = self.col_width if width is None else width
        # If eff_width is very large, this effectively disables wrapping for that cell.
        return textwrap.wrap(text, eff_width) or [""]

    def as_table(self) -> str:
        """
        Generate a formatted ASCII table with borders and aligned columns.

        Returns
        -------
        A string representation of the table.
        """
        # Step 0: Precompute dynamic widths for the first two columns.
        # Column 0 (branch): width = max length over header + all branch names
        branch_col_width = len(self.headers[0])  # "Model Branch Name"
        for branch in self.data.keys():
            branch_col_width = max(branch_col_width, len(str(branch)))

        # Column 1 (alias): width = max length over header + all alias strings (joined by ", \n")
        alias_col_width = len(self.headers[1])  # "Alias"
        for payload in self.data.values():
            alias_list = payload.get("alias", [])
            for alias in alias_list:
                alias_col_width = max(alias_col_width, len(str(alias)))

        # Step 1: Create raw rows (without wrapping)
        raw_rows: list[list[str]] = []
        # First row: header
        raw_rows.append(self.headers)

        # Data rows
        for branch, payload in self.data.items():
            alias_str = ", ".join(map(str, payload.get("alias", [])))
            info = payload.get("info") or {}
            row = [branch, alias_str] + [info.get(k, "") for k in self.info_keys]
            raw_rows.append(row)

        # Step 2: Wrap each cell, using dynamic widths for the first two columns,
        # and fixed `self.col_width` for info columns.
        wrapped_rows: list[list[list[str]]] = []
        for row in raw_rows:
            wrapped_row: list[list[str]] = []
            for j, cell in enumerate(row):
                if j == 0:
                    # First column: branch name -> no wrap by using its max width
                    wrapped_row.append(self._wrap_cell(cell, width=branch_col_width))
                elif j == 1:
                    # Second column: alias -> no wrap by using its max width
                    wrapped_row.append(self._wrap_cell(cell, width=alias_col_width))
                else:
                    # Info columns: keep using fixed col_width (wrapping allowed)
                    wrapped_row.append(self._wrap_cell(cell))
            wrapped_rows.append(wrapped_row)

        # Step 3: Determine actual width for each column
        # For the first two columns, we already decided the exact widths above.
        col_widths: list[int] = []
        for idx, col in enumerate(zip(*wrapped_rows)):
            if idx == 0:
                col_widths.append(branch_col_width)
            elif idx == 1:
                col_widths.append(alias_col_width)
            else:
                # Info columns: width is the maximum wrapped line length (<= self.col_width)
                col_widths.append(max(len(line) for cell in col for line in cell))

        # Helper: Draw a horizontal separator line
        def draw_separator() -> str:
            return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        # Helper: Draw one row of text parts (single lines per cell)
        def draw_row_line(cells_parts: list[list[str]]) -> str:
            return (
                "| "
                + " | ".join(
                    part.ljust(width) for part, width in zip(cells_parts, col_widths)
                )
                + " |"
            )

        # Step 4: Build the table string
        table_lines = []
        table_lines.append(draw_separator())

        for i, row_cells in enumerate(wrapped_rows):
            # Determine the maximum number of wrapped lines in this row
            max_lines = max(len(cell) for cell in row_cells)
            # Draw each wrapped line
            for line_idx in range(max_lines):
                line_parts = [
                    cell[line_idx] if line_idx < len(cell) else "" for cell in row_cells
                ]
                table_lines.append(draw_row_line(line_parts))
            table_lines.append(draw_separator())

        return "\n".join(table_lines)


# Example usage
if __name__ == "__main__":
    data = OrderedDict(
        {
            "Omat": {
                "alias": ["Default", "Materials"],
                "info": {
                    "observed-type": [
                        "H",
                        "He",
                        "Li",
                        "Be",
                        "B",
                        "C",
                        "N",
                        "O",
                        "F",
                        "Ne",
                        "Na",
                        "Mg",
                        "Al",
                        "Si",
                        "P",
                        "S",
                        "Cl",
                        "Ar",
                        "K",
                        "Ca",
                        "Sc",
                        "Ti",
                        "V",
                        "Cr",
                        "Mn",
                        "Fe",
                        "Co",
                        "Ni",
                        "Cu",
                        "Zn",
                        "Ga",
                        "Ge",
                        "As",
                        "Se",
                        "Br",
                        "Kr",
                        "Rb",
                        "Sr",
                        "Y",
                        "Zr",
                        "Nb",
                        "Mo",
                        "Tc",
                        "Ru",
                        "Rh",
                        "Pd",
                        "Ag",
                        "Cd",
                        "In",
                        "Sn",
                        "Sb",
                        "Te",
                        "I",
                        "Xe",
                        "Cs",
                        "Ba",
                        "La",
                        "Ce",
                        "Pr",
                        "Nd",
                        "Pm",
                        "Sm",
                        "Eu",
                        "Gd",
                        "Tb",
                        "Dy",
                        "Ho",
                        "Er",
                        "Tm",
                        "Yb",
                        "Lu",
                        "Hf",
                        "Ta",
                        "W",
                        "Re",
                        "Os",
                        "Ir",
                        "Pt",
                        "Au",
                        "Hg",
                        "Tl",
                        "Pb",
                        "Bi",
                        "Th",
                        "Pa",
                        "U",
                        "Np",
                        "Pu",
                        "Ac",
                    ],
                    "description": "OMat24 is a large-scale open dataset containing over 110 million DFT calculations "
                    "spanning diverse structures and compositions. It is designed to support AI-driven "
                    "materials discovery by providing broad and deep coverage of chemical space.",
                },
            },
        }
    )

    wrapper = OrderedDictTableWrapper(data, col_width=20)
    print(wrapper.as_table())  # noqa:T201
