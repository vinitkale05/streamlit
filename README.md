# Streamlit

[![GitHub stars](https://img.shields.io/github/stars/streamlit/streamlit.svg)](https://github.com/streamlit/streamlit/stargazers)
[![License](https://img.shields.io/github/license/streamlit/streamlit.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)

Streamlit — A faster way to build and share data apps.

Streamlit is an open-source app framework for Machine Learning and Data Science projects. It lets you turn data scripts into shareable web apps in minutes, all in pure Python. No front‑end experience required.

- **Homepage:** [https://streamlit.io](https://streamlit.io)
- **Repository:** [streamlit/streamlit](https://github.com/streamlit/streamlit)

---

## Key Features

- **Pure Python:** Build beautiful apps with minimal code.
- **Instant Deployment:** Share apps instantly with your team or the world.
- **Custom Widgets:** Interactive widgets out of the box for user input.
- **Component System:** Easily extend functionality with [custom components](https://docs.streamlit.io/develop/concepts/custom-components).
- **Integration:** Works with major data science libraries (NumPy, Pandas, Matplotlib, PyTorch, TensorFlow, etc).
- **Live Updates:** Automatic updating as you modify your code.
- **Caching:** Built-in caching for expensive computations.
- **Open Source:** Licensed under Apache 2.0.

---

## Installation

Streamlit requires Python 3.7 or later.

```bash
pip install streamlit
```

## Quickstart

Create a file `app.py`:

```python
import streamlit as st

st.title('Hello, Streamlit!')
st.write("Welcome to your first Streamlit app.")
x = st.slider('Select a value')
st.write(f'You selected: {x}')
```

Run your app:

```bash
streamlit run app.py
```

Open your browser to [http://localhost:8501](http://localhost:8501).

---

## Documentation

- [Getting Started](https://docs.streamlit.io/library/get-started)
- [API Reference](https://docs.streamlit.io/library/api-reference)
- [Component Library](https://docs.streamlit.io/develop/concepts/custom-components)
- [Community Forums](https://discuss.streamlit.io/)

---

## Developing Streamlit

### Frontend

The frontend is a monorepo of packages (app, connection, lib, protobuf, utils, typescript-config). To start developing:

```bash
yarn start
```
or
```bash
make frontend-dev
```

See [frontend/README.md](frontend/README.md) for details.

### Components

To create custom components, use [Component Template repo](https://github.com/streamlit/component-template).

---

## License

Streamlit is completely free and open-source and licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

---

## Contributing

We welcome contributions! Please see the [contributing guide](CONTRIBUTING.md) for more details.

---

## Acknowledgements

Streamlit is maintained by [Streamlit Inc.](https://streamlit.io) (now part of Snowflake Inc.) and a community of contributors.
