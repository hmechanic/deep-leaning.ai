# MCP

A deep learning project with Python notebooks and scripts, created as part of the deeplearning.ai MPC course.

## License

This project is licensed under the **MIT License**.

---

## Getting Started

### Prerequisites

- **Python**: Version `>=3.13` is required.
- **Poetry**: A tool for dependency management.

### Installation

1.  **Install Poetry**:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2.  **Configure Python version** (example using `pyenv`):
    ```bash
    pyenv install 3.13.0
    pyenv local 3.13.0
    ```
    *Note: You can use any Python version that meets the `>=3.13` requirement.*

3.  **Install dependencies**:
    ```bash
    poetry install --no-root
    ```

4.  **Set up Jupyter Kernel**:
    ```bash
    poetry run python -m ipykernel install --user --name mcp
    ```

5.  **Launch Jupyter Notebook**:
    ```bash
    poetry run jupyter notebook
    ```

---

### Core Dependencies

This project relies on the following main libraries:

- `anthropic`
- `arxiv`
- `pypdf2`
- `python-dotenv`
- `uv`
- `ipykernel`
- `notebook`
