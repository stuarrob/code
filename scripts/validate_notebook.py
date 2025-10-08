"""
Notebook Validation Script
Executes notebooks and checks for errors before deployment
"""

import subprocess
import sys
import os
from pathlib import Path
import json


def validate_notebook(notebook_path: str) -> tuple[bool, str]:
    """
    Execute a notebook and check for errors

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Tuple of (success: bool, message: str)
    """
    notebook_path = Path(notebook_path)

    if not notebook_path.exists():
        return False, f"Notebook not found: {notebook_path}"

    print(f"Validating notebook: {notebook_path.name}")
    print("=" * 60)

    # Execute the notebook (from project root)
    output_path = (
        notebook_path.parent / f"{notebook_path.stem}_validated{notebook_path.suffix}"
    )

    # Change to project root before executing
    project_root = Path.cwd()

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.kernel_name=etftrader",
        "--ExecutePreprocessor.cwd=/home/stuar/code/ETFTrader",
        "--output",
        str(output_path.absolute()),
        str(notebook_path.absolute()),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd="/home/stuar/code/ETFTrader",  # Run from project root
        )

        if result.returncode == 0:
            print(f"✅ Notebook executed successfully!")
            print(f"   Output saved to: {output_path}")

            # Clean up the validated notebook if successful
            if output_path.exists():
                output_path.unlink()

            return True, "Notebook validation passed"
        else:
            print(f"❌ Notebook execution failed!")
            print(f"\nError output:")
            print(result.stderr)
            return False, f"Execution failed:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Notebook execution timed out (>5 minutes)"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def check_notebook_structure(notebook_path: str) -> tuple[bool, str]:
    """
    Check notebook structure and metadata

    Args:
        notebook_path: Path to the notebook file

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        with open(notebook_path, "r") as f:
            nb = json.load(f)

        # Check for required kernel
        kernel_name = nb.get("metadata", {}).get("kernelspec", {}).get("name")

        issues = []

        if not kernel_name:
            issues.append("No kernel specified")
        elif kernel_name not in ["etftrader", "python3"]:
            issues.append(f"Unexpected kernel: {kernel_name}")

        # Check for empty cells
        empty_code_cells = sum(
            1
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code" and not cell.get("source")
        )

        if empty_code_cells > 0:
            issues.append(f"{empty_code_cells} empty code cells")

        if issues:
            return False, "Structure issues: " + ", ".join(issues)
        else:
            return True, "Structure validation passed"

    except Exception as e:
        return False, f"Structure check error: {str(e)}"


def main():
    """Main validation function"""
    if len(sys.argv) < 2:
        print("Usage: python validate_notebook.py <notebook_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]

    print("\n" + "=" * 60)
    print("NOTEBOOK VALIDATION")
    print("=" * 60 + "\n")

    # Step 1: Structure check
    print("Step 1: Checking notebook structure...")
    success, message = check_notebook_structure(notebook_path)
    print(f"  {message}\n")

    if not success:
        print("❌ Validation failed at structure check")
        sys.exit(1)

    # Step 2: Execution check
    print("Step 2: Executing notebook...")
    success, message = validate_notebook(notebook_path)
    print(f"  {message}\n")

    if not success:
        print("❌ Validation failed at execution")
        sys.exit(1)

    print("=" * 60)
    print("✅ ALL VALIDATIONS PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
