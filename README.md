# KreinSynergy: Interactive Exploration of Indefinite Kernels for Antibiotic Synergy Research
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16547498.svg)](https://doi.org/10.5281/zenodo.16547498)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![GitHub release](https://img.shields.io/github/v/release/mojo8787/LearningKernel.svg)](https://github.com/mojo8787/LearningKernel/releases)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0003--2070--2811-green.svg)](https://orcid.org/0000-0003-2070-2811)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

## About

KreinSynergy is an interactive educational tool that demonstrates how Kreĭn space kernels can be applied to antibiotic synergy research. The application bridges the gap between machine learning theory and microbiology applications, providing an accessible interface for researchers to explore the mathematical concepts behind indefinite kernels and their biological applications.

## Author

**Dr. Almotasem Bellah, PhD**  
Computational Biologist / Microbiologist  
Brno, Czech Republic  
Email: motasem.youniss@gmail.com

## Features

- **Home**: Overview of Kreĭn space kernels and their relevance to antibiotic research
- **Kernel Types Overview**: Detailed explanation of different kernel types and their properties
- **Kernel Visualization**: Interactive visualization of kernel transformations with various data types
- **Kreĭn Space Mathematics**: In-depth explanation of the mathematical principles with interactive demos
- **Biological Context**: Models and visualizations of antibiotic synergy mechanisms
- **Experimental Data**: Analysis of antibiotic combination data from the integrated PostgreSQL database
- **Save & Load Analysis**: Functionality to save analysis results for future reference

## Mathematical Foundation

Kreĭn space kernels are a class of indefinite kernels that can be represented as the difference of two positive definite kernels:

```
K(x, y) = K₊(x, y) - K₋(x, y)
```

Where K₊ and K₋ are positive definite kernels representing similarity and dissimilarity components, respectively. This structure makes them ideal for modeling complex biological interactions that involve both cooperative and competitive effects.

## Biological Relevance

Antibiotic synergy occurs when two antibiotics combined have a greater effect than the sum of their individual effects. This interaction can be modeled using Kreĭn kernels, which can capture both:

- **Synergistic interactions**: When drugs enhance each other's effects
- **Antagonistic interactions**: When drugs interfere with each other's effects
- **Complex pathway interactions**: With both cooperative and competitive components

## Technical Implementation

The application is built with:

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms and kernel implementations
- **Plotly**: Interactive data visualizations
- **SQLAlchemy**: Database ORM for PostgreSQL integration
- **PostgreSQL**: Database for storing antibiotic data and analysis results

## Research Context

This project is inspired by the work of Professor Thomas Gärtner on learning with indefinite kernels and their applications in computational biology. It serves as a demonstration of the potential for applying advanced kernel methods to challenging problems in microbiology and drug discovery.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up a PostgreSQL database and configure environment variables
4. Initialize the database: `python init_db.py`
5. Run the application: `streamlit run app.py`

## Future Directions

- Integration with real experimental data from antibiotic sensitivity testing
- Expansion to include more complex biological network models
- Implementation of additional kernel methods specific to molecular data
- Development of predictive models for antibiotic combination efficacy

## References

- Ong, C.S., Mary, X., Canu, S., Smola, A.J.: Learning with non-positive kernels. In: Proceedings of the Twenty-First International Conference on Machine Learning (2004)
- Haasdonk, B.: Feature space interpretation of SVMs with indefinite kernels. IEEE Transactions on Pattern Analysis and Machine Intelligence (2005)
- Gärtner, T., Lloyd, J.W., Flach, P.A.: Kernels and distances for structured data. Machine Learning (2004)

## License

This project is licensed under the MIT License.