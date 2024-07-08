# Classification Tree Optimization Project

## Course Information

This project is part of the Master's Degree in Computer Engineering course, specifically for the "Optimization" class taught by Professor Francesca Maggioni.

## Assignment

The goal of this project is to **implement one of the classification trees** presented in the paper by Bertsimas and Dunn and apply it to a real-life dataset discussed in the article.

### Paper Reference

- **Title**: Optimal Classification Trees
- **Authors**: Dimitris Bertsimas and Julia Dunn
- **Journal**: Machine Learning
- **Volume**: 106
- **Pages**: 1039–1082
- **Year**: 2017
- **DOI**: [10.1007/s10994-017-5633-9](https://doi.org/10.1007/s10994-017-5633-9)

## Setup Instructions

To ensure a smooth development environment, it is recommended to use a virtual environment. You can use either `venv` or `conda`. Ensure you are using **Python 3.12** for compatibility.

### Using `venv`

1. **Create a virtual environment**:
    ```bash
    python -m venv myenv
    ```

2. **Activate the virtual environment**:
    - On Windows:
      ```bash
      myenv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source myenv/bin/activate
      ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Using `conda`

1. **Create a new conda environment** with Python 3.12:
    ```bash
    conda create -n myenv python=3.12
    ```

2. **Activate the conda environment**:
    ```bash
    conda activate myenv
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

The required dependencies are listed in `requirements.txt`. Make sure to install them using one of the methods described above.

## Usage

1. **Request key**:
    Ensure you have the key to use a solver like Gurobi. You can request an academic license from the [Gurobi website](https://www.gurobi.com/academia/academic-program-and-licenses/).

    You can also use other open-source solvers like `ipopt`.

2. **Run the Notebook**:
    Open the Jupyter notebook and follow the steps to load data, configure the model, train it, and evaluate performance.

## Contribution

Feel free to fork the repository and submit pull requests. If you encounter any issues or have suggestions, please open an issue in the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.