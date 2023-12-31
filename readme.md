# LangChain RAG

LangChain RAG Tips and Tricks, including Self Query

Based on and inspired by YouTube videos by [Sam Witteveen](https://www.youtube.com/watch?v=f4LeWlt3T8Y&ab_channel=SamWitteveen)

## Installation

Install [Langchain](https://github.com/hwchase17/langchain) and other required packages, some packages are mentioned inside the scripts.

```
pip install langchain huggingface_hub openai google-search-results tiktoken chromadb lark prettytable termcolor
```

Using the Together API

```
pip install --upgrade together
```

Modify `constants.py.default` to use your own [OpenAI API key](https://platform.openai.com/account/api-keys), and rename it to `constants.py`.

## Example usage

```
> python main.py
```
