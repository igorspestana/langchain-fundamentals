## Principais recursos do LangChain

- Criação de chains: 
    - Fluxo de execução compostos por etapas (runnables). 
    - Cada etapa pode chamar modelos, processar dados, etc. 
    - São usados para tarefas complexas que requerem tomada de decisão. 
    - São usados para processar dados de forma sequencial.
- Langchain Expression Language (LCEL): 
    - Realiza essas operações utilizando "pipe" "|" para compor runnables.
    - São usados para compor runnables de forma sequencial.
- Carregamento e divisão de documentos: 
    - Consegue carregar CSVs, JSON, HTML, Markdown, Sites. 
    - É possível dividir documentos em partes menores para processamento. 
    - É possível carregar documentos de diferentes formatos.
- Modelos de Embedding e Armazenamento Vetorial: 
    - Interface para gerar embeddings com OpenAI, HuggingFace, etc 
    - Armazenar vetores em bancos como Pinecone, PGVector (postgres), Weaviate, FAISS... 
    - São usados para buscar similaridade em textos.
- Busca por similaridade em bancos de dados vetoriais: 
    - Busca por similaridade em bancos de dados vetoriais.
    - São usados para buscar similaridade em textos.
- Agentes e Ferramentas: 
    - Agentes usam a LLM para processar a entrada e conseguem decidir quais ações / ferramentas executar. 
    - São usados para tarefas complexas que requerem tomada de decisão. 
    - Ferramentas são usadas para executar ações / tarefas específicas.
- Memória e histórico: 
    - Memória para agentes e histórico para modelos de LLM. 
    - Memória é usado para armazenar informações relevantes para o processamento da tarefa.
- Prompt Templates e placeholders: 
    - Templates são usados para gerar prompts para modelos de LLM. 
    - Placeholders são usados para substituir partes do prompt por valores dinâmicos.
    - PromptTemplate: template básico com variáveis de entrada
    - ChatPromptTemplate: template para conversas com múltiplos tipos de mensagem
    - Métodos: format(), invoke(), format_messages(), partial()
    - Vantagens: reutilização, manutenção, flexibilidade, integração
- OutputParsing e Pydantic: 
    - Parseamento de saída para formatos específicos,如 JSON
- Sumarização, utilizando map-reduce: 
    - Sumarização de textos longos em textos mais curtos.
- Runnables: 
    - Componentes executáveis que podem ser conectados em pipelines usando o LangChain Expression Language (LCEL).
    - Representam operações que podem ser executadas como parte de uma chain.

## Instalação

```
python -m venv venv
```
```
source venv/bin/activate
```

```
pip install langchain langchain-openai langchain-google-genai langchain-google-vertexai python-dotenv beautifulsoup4 pypdf
```

```
pip freeze > requirements.txt
```

### Configuração do ambiente

Crie um arquivo `.env` na raiz do projeto com suas chaves de API:

```
# OpenAI API Key
OPENAI_API_KEY=sua_chave_openai_aqui

# Google API Key (opcional)
GOOGLE_API_KEY=sua_chave_google_aqui
```

### Comandos úteis do ambiente virtual

```
# Ativar ambiente virtual
source venv/bin/activate

# Desativar ambiente virtual
deactivate
```

### Dependências explicadas:

- **langchain**: Framework principal para desenvolvimento de aplicações com LLMs. Fornece abstrações para chains, agentes, memória, prompts e integração com diferentes provedores de modelos.

- **langchain-openai**: Integração específica com modelos da OpenAI (GPT-3.5, GPT-4, etc.). Permite usar os modelos da OpenAI diretamente nas chains e agentes do LangChain.

- **langchain-google-genai**: Integração com modelos do Google (Gemini, PaLM). Fornece acesso aos modelos de IA do Google através do LangChain.

- **langchain-google-vertexai**: Integração com Google Vertex AI. Permite usar os modelos do Google Cloud Platform através do LangChain, incluindo funcionalidades avançadas como embeddings e busca vetorial.

- **python-dotenv**: Biblioteca para carregar variáveis de ambiente de arquivos `.env`. Essencial para gerenciar chaves de API e configurações sensíveis sem expô-las no código.

- **beautifulsoup4**: Parser HTML/XML para extrair e processar conteúdo de páginas web. No LangChain, é usado pelos loaders de documentos para converter HTML em texto limpo para processamento por LLMs. Também é amplamente utilizado para web scraping em geral.

- **pypdf**: Biblioteca para manipular arquivos PDF. Permite extrair texto, metadados e outras informações de documentos PDF para processamento com LLMs.


## Diferenças entre abordagens de inicialização de modelos

### ChatOpenAI direto vs init_chat_model

#### Abordagem 1: ChatOpenAI direto
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
```

**Vantagens:**
- Simples e direto
- Melhor suporte a recursos específicos da OpenAI
- Menos overhead de performance
- Tipagem mais específica
- Acesso a parâmetros específicos da OpenAI

**Desvantagens:**
- Acoplado à OpenAI
- Troca de provedor exige mudança de código
- Menos flexível

#### Abordagem 2: init_chat_model
```python
from langchain.chat_models import init_chat_model
model = init_chat_model(model="gpt-4o-mini", temperature=0.5, model_provider="openai")
```

**Vantagens:**
- Flexível para múltiplos provedores
- Troca de provedor via configuração
- Abstração consistente entre provedores
- Útil para testes com diferentes modelos

**Desvantagens:**
- Mais parâmetros necessários
- Menos recursos específicos por provedor
- Possível overhead de performance
- Menos tipagem específica

### Quando usar cada abordagem

**Use ChatOpenAI direto quando:**
- Projeto focado na OpenAI
- Precisa de recursos específicos da OpenAI
- Performance é crítica
- Não há planos de trocar de provedor
- Desenvolvimento rápido

**Use init_chat_model quando:**
- Precisa de flexibilidade entre provedores
- Desenvolvimento de biblioteca/framework
- Testes com múltiplos modelos
- Configuração dinâmica de provedor
- Projeto que pode migrar entre provedores

### Recomendação
Para a maioria dos casos, prefira **ChatOpenAI direto** pela simplicidade e performance. Use **init_chat_model** quando a flexibilidade entre provedores for necessária.

## Prompt Templates

### PromptTemplate Básico

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hello, {name}!",
)

# Métodos de formatação
print(template.format(name="John"))  # Hello, John!
print(template.invoke({"name": "John"}))  # Hello, John!
```

**Características:**
- `input_variables`: lista de variáveis do template
- `template`: string com placeholders `{variavel}`
- `format()`: substitui variáveis e retorna string
- `invoke()`: substitui variáveis e retorna PromptValue

### ChatPromptTemplate

```python
from langchain.prompts import ChatPromptTemplate

system_prompt = ("system", "You are a helpful assistant that can answer questions in {language}.")
user_prompt = ("user", "{question}")

chat_prompt_template = ChatPromptTemplate([system_prompt, user_prompt])

messages = chat_prompt_template.format_messages(
    language="English", 
    question="What is the capital of France?"
)

for message in messages:
    print(f"{message.type}: {message.content}")
```

**Características:**
- Suporta múltiplos tipos de mensagem (system, user, assistant)
- Cada mensagem é uma tupla `(tipo, template)`
- `format_messages()`: retorna lista de objetos Message
- Útil para conversas estruturadas

### Vantagens dos Prompt Templates

**Reutilização:**
- Mesmo template para diferentes entradas
- Evita duplicação de código

**Manutenção:**
- Mudanças centralizadas
- Versionamento simples

**Flexibilidade:**
- Placeholders dinâmicos
- Suporte a múltiplos tipos de mensagem

**Integração:**
- Compatível com chains e agentes
- Funciona com diferentes provedores de LLM

### Métodos Principais

- `format(**kwargs)`: retorna string formatada
- `invoke(input_dict)`: retorna PromptValue
- `format_messages(**kwargs)`: retorna lista de Message (ChatPromptTemplate)
- `partial(**kwargs)`: cria template com valores pré-definidos

### Casos de Uso Comuns

**Templates simples:**
```python
template = PromptTemplate(
    input_variables=["product", "price"],
    template="The {product} costs ${price}."
)
```

**Templates complexos:**
```python
template = PromptTemplate(
    input_variables=["context", "question", "format"],
    template="""
    Context: {context}
    
    Question: {question}
    
    Please answer in {format} format.
    """
)
```

**Chat templates:**
```python
chat_template = ChatPromptTemplate([
    ("system", "You are a {role} assistant."),
    ("user", "Help me with: {task}"),
    ("assistant", "I'll help you with that."),
    ("user", "{follow_up}")
])
```

## LangChain Expression Language (LCEL) e o Decorator @chain

### O que é o decorator @chain

O decorator `@chain` é uma função decorator do LangChain que converte funções Python comuns em objetos `Runnable`. Estes objetos podem ser usados diretamente em chains e pipelines usando o LangChain Expression Language (LCEL).

```python
from langchain_core.runnables import chain

@chain
def square(x: int) -> dict:
    return {"square_result": x * x, "x": x}
```

### Como funciona

1. **Conversão para Runnable**: O decorator pega sua função Python e a transforma em um objeto que implementa a interface `Runnable`
2. **Compatibilidade com LCEL**: Funciona nativamente com o operador pipe `|` do LangChain
3. **Processamento automático**: Gerencia automaticamente entrada e saída de dados
4. **Integração com pipelines**: Pode ser combinado com outros componentes (prompts, modelos, etc.)

### Características principais

**Entrada de dados:**
- Aceita dados diretamente (int, str, dict) ou através de invocação via `.invoke()`
- Processamento automático de tipos de entrada

**Saída de dados:**
- Retorna dados estruturados que podem ser passados para o próximo componente
- Mantém compatibilidade de tipos para o pipeline

**Flexibilidade:**
- Pode receber dados de diferentes formas
- Suporte a tipos primitivos e dicionários

### Exemplos de uso

**Exemplo básico:**
```python
@chain
def square(x: int) -> dict:
    return {"square_result": x * x, "x": x}

# Uso direto
result = square.invoke(4)
print(result)  # {'square_result': 26, 'x': 4}

# Uso em pipeline
pipeline = square | question_template | model
response = pipeline.invoke(4)
```

**Exemplo com entrada como dicionário:**
```python
@chain
def process_user(input_dict: dict) -> dict:
    name = input_dict["name"]
    age = input_data["age"]
    return {
        "greeting": f"Hello {name}!",
        "age_group": "adult" if age >= 18 else "minor"
    }

result = process_user.invoke({"name": "Alice", "age": 25})
```

**Pipeline completo:**
```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain

@chain
def square(x: int) -> dict:
    return {"square_result": x * x, "x": x}

question_template = PromptTemplate(
    input_variables=["x"],
    template="What is the square of {x}? Is it {square_result}?",
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Conceito importante: o pipe | conecta os componentes
pipeline = square | question_template | model
response = pipeline.invoke(4)
```

### Vantagens do decorator @chain

**Simplicidade:**
- Converte funções Python regulares em componentes executáveis
- Não precisa herdar classes ou implementar interfaces complexas

**Reutilização:**
- Funções podem ser usadas independentemente ou em pipelines
- Fácil composição com outros componentes

**Flexibilidade:**
- Aceita diferentes tipos de entrada
- Compatível com todo o ecossistema LangChain

**Performance:**
- Processamento eficiente de dados
- Integração nativa com o sistema de caching do LangChain

### Quando usar

**Use @chain quando:**
- Tem uma função que processa dados e quer integrá-la ao LangChain
- Precisa criar componentes personalizados para seu pipeline
- Quer converter lógica Python existente em runnables
- Precisa de flexibilidade na entrada de dados

**Não use @chain quando:**
- A função é muito simples e pode ser feita diretamente no pipeline
- Precisa de lógica muito complexa de estado ou controle de fluxo
- Está trabalhando apenas com templates simples (use PromptTemplate)

### Diferenças importantes

**@chain vs PromptTemplate:**
- `@chain`: Para lógica de processamento personalizada
- `PromptTemplate`: Para formatação de texto com placeholders

**@chain vs Lambda/LambdaChain:**
- `@chain`: Função nomeada decorata, mais legível e reutilizável
- `Lambda`: Função anônima inline, menos legível mas mais concisa

**Integração com LCEL:**
- O decorator `@chain` é especificamente projetado para trabalhar com o operador pipe `|`
- Permite composição fluida e expressiva de componentes
- Mantém o estilo funcional do LangChain Expression Language

## O que são Runnables

### Definição

Runnables são componentes do LangChain que implementam uma interface padrão para execução. Eles representam qualquer operação que pode ser executada como parte de uma pipeline ou chain, desde modelos de linguagem até funções de processamento de dados.

### Interface Runnable

Todo runnable deve implementar pelo menos um dos seguintes métodos:

**Métodos principais:**
- `invoke(input)`: Executa o componente com uma entrada e retorna resultado singular
- `ainvoke(input)`: Versão assíncrona do `invoke()`
- `batch(inputs)`: Executa múltiplas entradas em paralelo e retorna lista de resultados
- `abatch(inputs)`: Versão assíncrona do `batch()`
- `stream(input)`: Stream de resultados em tempo real
- `astream(input)`: Versão assíncrona do `stream()`

### Tipos de Runnables

**1. Modelos (Models):**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello, world!")
```

**2. Prompts:**
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)
formatted = prompt.invoke({"topic": "machine learning"})
```

**3. Funções decoradas:**
```python
from langchain_core.runnables import chain

@chain
def processor(x: int) -> dict:
    return {"processed": x * 2, "original": x}

result = processor.invoke(5)
```

**4. Chains (sequências de runnables):**
```python
chain = prompt | llm
response = chain.invoke({"topic": "AI"})
```

**5. Output Parsers:**
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
clean_text = parser.invoke("Raw LLM response text")
```

### Características dos Runnables

**Uniformidade:**
- Todos os runnables seguem a mesma interface
- Métodos consistentes (`invoke`, `batch`, `stream`)
- Compatibilidade de entrada/saída

**Composição:**
- Podem ser conectados usando o operador pipe `|`
- Suporte a pipelines complexos
- Reutilização de componentes

**Flexibilidade:**
- Assíncronos e síncronos
- Suporte a streaming e batch processing
- Configuração de paralelismo

### Exemplos de Composição

**Pipeline simples:**
```python
# Cada componente é um runnable
pipeline = prompt_template | llm | output_parser
result = pipeline.invoke({"topic": "Python"})
```

**Pipeline complexo com paralelização:**
```python
from langchain_core.runnables import RunnableParallel

# Execute múltiplos componentes em paralelo
parallel_pipeline = RunnableParallel(
    llm_response=prompt | llm,
    summaries=summarizer | llm,
    keywords=keyword_extractor | llm
)

results = parallel_pipeline.invoke({"document": text})
```

### Vantagens dos Runnables

**Interoperabilidade:**
- Qualquer componente pode trabalhar com qualquer outro
- Facilita migração e substituição de componentes

**Escalabilidade:**
- Suporte nativo a processamento paralelo
- Stream processing para grandes volumes de dados

**Manutenibilidade:**
- Interface consistente facilita debugging
- Componentes testáveis independentemente

**Performance:**
- Otimizações automáticas de pipeline
- Cache inteligente entre componentes

### Quando usar Runnables

**Use runnables quando:**
- Construindo pipelines ou chains complexas
- Precisa de interoperabilidade entre componentes
- Quer aproveitar paralelização automática
- Desenvolve bibliotecas ou frameworks extensíveis

**Runnables são menos necessários quando:**
- Trabalhando com casos simples de uso único
- Não precisa de composição ou reutilização
- Performance não é crítica
- Pipeline muito específico e estático

## RunnableLambda: Convertendo Funções em Runnables

### O que é RunnableLambda

`RunnableLambda` é uma classe do LangChain que permite envolver qualquer função Python em um objeto runnable. É uma alternativa mais explícita ao decorator `@chain`, especialmente útil quando você já tem funções existentes que quer integrar ao LangChain sem modificá-las.

### Diferenças entre @chain e RunnableLambda

**Decorator @chain:**
- Modifica a função original
- Mais conciso e limpo
- Sintaxis especializada para LangChain

**RunnableLambda:**
- Preserva a função original intacta
- Mais explícito na conversão
- Permite envolver funções existentes sem modificação

### Exemplo básico

**Função existente:**
```python
def square(x: int) -> dict:
    return {"square_result": x * x, "x": x}

# Envolvendo em RunnableLambda
square_lambda = RunnableLambda(square)

# Agora pode ser usado em pipelines
pipeline = square_lambda | prompt_template | model
response = pipeline.invoke(4)
```

**Comparação com decorator:**
```python
# Com @chain
@chain
def square_chain(x: int) -> dict:
    return {"square_result": x * x, "x": x}

# Com RunnableLambda
def square_func(x: int) -> dict:
    return {"square_result": x * x, "x": x}

square_runnable = RunnableLambda(square_func)
```

### Casos de uso típicos

**1. Integração de funções legadas:**
```python
# Função já existente no código
def process_data(data: dict) -> dict:
    processed = data.copy()
    processed["timestamp"] = datetime.now().isoformat()
    processed["checksum"] = calculate_checksum(data["content"])
    return processed

# Transformar em runnable sem alterar original
processor = RunnableLambda(process_data)

pipeline = processor | formatter | llm
result = pipeline.invoke({"content": "example data"})
```

**2. Funções de transformação simples:**
```python
def clean_text(text: str) -> str:
    return text.strip().lower()

def extract_keywords(text: str) -> list:
    return text.split()

# Múltiplas funções em pipeline
cleaner = RunnableLambda(clean_text)
extractor = RunnableLambda(extract_keywords)

pipeline = cleaner | extractor | keyword_processor | llm
```

**3. Bridge entre código existente e LangChain:**
```python
def validate_input(data: dict) -> dict:
    if "required_field" not in data:
        data["required_field"] = "default_value"
    return data

def transform_for_llm(data: dict) -> str:
    return f"Process: {data['required_field']} -> {data.get('options', [])}"

validator = RunnableLambda(validate_input)
transformer = RunnableLambda(transform_for_llm)

chain = validator | transformer | prompt | llm | parser
```

### Vantagens do RunnableLambda

**Flexibilidade:**
- Não modifica a função original
- Permite teste independente da função base
- Facilita refatoração gradual

**Compatibilidade:**
- Funciona com qualquer função callable
- Suporte a lambdas inline
- Integração com bibliotecas externas

**Clareza:**
- Torna explícita a conversão para runnable
- Facilita debugging e rastreamento
- Separação clara entre lógica e infraestrutura

### Exemplos avançados

**Lambda inline:**
```python
# Lambda function direta
adder = RunnableLambda(lambda x: x + 1)
multiplier = RunnableLambda(lambda x: x * 2)

# Pipeline funcional
pipeline = adder | multiplier | formatter
result = pipeline.invoke(5)  # Resultado: 12
```

**Funções com múltiplos parâmetros:**
```python
def complex_processor(arg1: str, arg2: int, arg3: bool) -> dict:
    return {
        "processed": arg1 * arg2 if arg3 else arg1,
        "metadata": {"arg2": arg2, "arg3": arg3}
    }

# Como RunnableLambda precisa de uma função que aceite um parâmetro
def wrapper(data: dict) -> dict:
    return complex_processor(data["arg1"], data["arg2"], data["arg3"])

processor = RunnableLambda(wrapper)
```

**Funções de bibliotecas externas:**
```python
import pandas as pd

def analyze_dataframe(df_dict: dict) -> dict:
    df = pd.DataFrame(df_dict["data"])
    return {
        "mean": df.mean().to_dict(),
        "count": df.shape[0],
        "columns": list(df.columns)
    }

analyzer = RunnableLambda(analyze_dataframe)
analysis_pipeline = analyzer | summary_generator | llm
```

### Quando usar RunnableLambda vs @chain

**Use RunnableLambda quando:**
- Você tem funções existentes que não quer modificar
- Precisa de clareza explícita na conversão para runnable
- Quer manter a função original para outros propósitos
- Está integrando código legado/third-party ao LangChain

**Use @chain quando:**
- Criando novas funções específicas para LangChain
- Quer código mais conciso
- A função será usada exclusivamente em pipelines
- Está seguindo padrões específicos do LangChain

### Limitações e considerações

**Limitações:**
- Função deve aceitar pelo menos um parâmetro
- Comportamento de chamada deve ser previsível
- Não suporta funções com `*args`, `**kwargs` dinâmicos facilmente

**Considerações de performance:**
- Overhead mínimo de conversão
- Cache automático do LangChain aplicado
- Streaming funcionará se a função suportar

**Padrões recomendados:**
- Use tipo hints para melhor debugging
- Teste a função isoladamente antes de envolver
- Documente comportamentos esperados de entrada/saída