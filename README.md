# Personal AI Data Analyst

A local, privacy-focused AI data analyst application built with Python, Streamlit, and Ollama.

## Features

- **Privacy First**: Runs entirely on your local machine. No data is sent to the cloud.
- **Interactive Dashboard**: Upload CSV, Excel, or JSON files and interact with your data.
- **AI-Powered Analysis**: Uses local LLMs (via Ollama) to interpret natural language queries and generate Python code for analysis.
- **Deterministic Suggestions**: Automatically suggests relevant questions and visualizations based on your data structure.
- **Visualizations**: Generates histograms, scatter plots, correlation matrices, and time series plots.

## Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: [Download and install Ollama](https://ollama.com/).
3.  **Llama 3.1 Model**: Pull the model using the command:
    ```bash
    ollama pull llama3.1
    ```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd personal-ai-data-analyst
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2.  Open your browser at `http://localhost:8501`.

3.  Upload a dataset and start asking questions!

## Technologies

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Ollama](https://ollama.com/)

## License

MIT
