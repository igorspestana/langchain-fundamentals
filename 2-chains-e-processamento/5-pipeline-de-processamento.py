"""
PIPELINE DE PROCESSAMENTO COM LANGCHAIN

Este exemplo demonstra como criar pipelines complexos onde o resultado de uma chain
serve como entrada para outra chain, criando um fluxo sequencial de processamento.

CASOS DE USO PRÁTICOS:
- Documento técnico → Traduzir → Resumir para executivos
- Artigo científico → Extrair pontos-chave → Gerar conteúdo educativo  
- Feedback de usuário → Traduzir → Analisar sentimento
- Contrato jurídico → Simplificar linguagem → Extrair obrigações

FLUXO DO PIPELINE:
┌─────────────────┐    ┌─────────────┐    ┌──────────────────┐
│   Input Text    │ -> │ Translate   │ -> │ English Text     │
│   (Português)   │    │ to English  │    │                  │
└─────────────────┘    └─────────────┘    └──────────────────┘
                                                       |
                                                       v
┌─────────────────────┐    ┌─────────────┐    ┌─────────────────┐
│  Final Summary      │ <- │ Summarize   │ <- │  Text Input     │
│  (English)          │    │ Translation │    │ (rekeys to      │
└─────────────────────┘    └─────────────┘    │  "text" field)  │
                                               └─────────────────┘
"""

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["input_text"],
    template="Translate the following text to English: {input_text}",
)

template_summarize = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 5 words: {text}",
)

llm_english = ChatOpenAI(model="gpt-4o-mini", temperature=0)

translate_chain = template_translate | llm_english | StrOutputParser()
pipeline = {"text": translate_chain} | template_summarize | llm_english | StrOutputParser()

input_data = {"input_text": "LangChain é uma biblioteca de código aberto para construir aplicações com modelos de linguagem."}

response = pipeline.invoke(input_data)
print("Pipeline Result:")
print(f"Input: {input_data['input_text']}")
print(f"Output: {response}")