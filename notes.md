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

criar ambiente virtual
```
python -m venv venv
```

ativar ambiente virtual
```
source venv/bin/activate
```

instalar dependências
```
pip install langchain langchain-openai langchain-google-genai langchain-google-vertexai python-dotenv beautifulsoup4 pypdf
```

caso tenha um requirements.txt instalar as dependências
```
pip install -r requirements.txt
```

caso queira criar um requirements.txt para instalar as dependências
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

## Sumarização de texto

### Sumarização de texto com LangChain

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```
O que é um chunk?
É uma parte do texto que é processada como uma unidade.

O que é um overlap?
É a sobreposição entre chunks.

chunk_size: tamanho do chunk
chunk_overlap: overlap entre chunks

É necessário fazer split do texto?
Sim, quando o texto pode ultrapassar o limite de contexto do modelo. O split permite processar partes menores e depois combinar as respostas.

Seria possível ou recomendável não fazer split do texto em algum caso?
- Sim, quando o documento for curto o suficiente para caber confortavelmente no contexto do modelo com folga para o prompt e instruções. Como regra prática: se o total de tokens de entrada + saída esperada ficar bem abaixo do limite do modelo, prefira não dividir (menos latência, menor custo e menos risco de inconsistências entre partes).
- Evite dividir quando a coesão global for essencial e a perda de contexto entre partes puder prejudicar a qualidade (por exemplo, resumos de artigos curtos, emails, issues pequenas, páginas breves).

Quando devo necessariamente dividir?
- Quando o documento exceder o limite de contexto do modelo.
- Quando você quiser paralelizar o processamento para reduzir latência em coleções grandes.
- Quando desejar controle sobre granularidade (ex.: sumarizar por seções/capítulos) ou aplicar técnicas como map_reduce/refine.

Como escolher chunk_size e chunk_overlap?
- chunk_size: escolha de forma a preservar unidades semânticas (parágrafos/seções). Valores comuns: 800–1500 caracteres para texto geral; ajuste por idioma e densidade de conteúdo. Para modelos com janelas maiores, você pode subir para 2000–4000 caracteres se desejar maior coesão por chunk.
- chunk_overlap: 10%–20% do chunk_size costuma ser suficiente. Aumente o overlap quando há conceitos que atravessam parágrafos, reduza quando o texto é bem modular.
- Prefira splitters que respeitem sentenças e parágrafos para minimizar cortes no meio de ideias.

### RecursiveCharacterTextSplitter vs CharacterTextSplitter

Qual é diferença entre RecursiveCharacterTextSplitter e CharacterTextSplitter?
- CharacterTextSplitter: corta por comprimento fixo de caracteres, sem olhar estrutura. É simples e rápido, mas pode quebrar sentenças e parágrafos no meio.
- RecursiveCharacterTextSplitter: tenta dividir recursivamente usando uma lista de separadores (por exemplo, "\n\n" para parágrafos, depois "\n" para linhas, depois ". " para sentenças, etc.). Mantém melhor a coesão semântica e geralmente produz chunks mais naturais.

Outros splitters úteis
- TokenTextSplitter: divide por tokens (aproxima melhor a janela do modelo).
- MarkdownTextSplitter: respeita a hierarquia de títulos/listas em Markdown.
- HTML/Text splitters específicos: preservam blocos semânticos (tags, seções).

### Diferença entre stuff e map_reduce

stuff:
- Usa o modelo para gerar o resumo completo do texto.
- É mais rápido e eficiente.
- Limitação: depende da janela de contexto (só funciona bem se todo material couber no prompt com folga).

map_reduce:
- Usa o modelo para gerar o resumo de cada chunk e depois combina os resumos.
- Escala para documentos muito longos, pois processa em partes.
- Permite controlar a estratégia de combinação (reduce) e garantir cobertura ampla do conteúdo.
- Tipicamente mais caro e mais lento do que stuff, por envolver múltiplas chamadas (uma por chunk + uma ou mais para redução).

O map_reduce é sempre mais caro que stuff?
- Em geral, sim. O custo cresce aproximadamente com o número de chunks processados, mais uma etapa de redução. Já o stuff faz 1 chamada, mas só se tudo couber no contexto. Se você puder usar stuff, ele é mais barato/rápido; se não couber, use map_reduce (ou refine) para escalar.

Qual é mais preciso?
- Depende do caso. Em textos curtos, stuff costuma ir melhor por manter o contexto global completo. Em textos longos, map_reduce pode ser mais robusto, pois garante que cada parte seja considerada antes da combinação. A qualidade final depende muito do prompt e da etapa de redução.

Boas práticas para sumarização
- Defina objetivo claro: extrativa (fatos) vs. abstrativa (reescrita) vs. tópica (bullet points, decisões, tarefas).
- Dê formato de saída explícito: bullets, JSON, parágrafos, limite de palavras/caracteres.
- Controle estilo: neutro, técnico, para leigos, com/sem citações.
- Use temperatura baixa (0.0–0.3) para resumos factuais e consistentes.
- Adicione instruções para evitar alucinações e citar trechos ou seções quando apropriado.
- Em map_reduce, seja explícito na etapa de reduce sobre como combinar, deduplicar e priorizar informações.

Exemplos práticos

Exemplo 1: Sumarização com stuff (documento curto)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

text = """Seu texto curto que cabe no contexto do modelo..."""

prompt = PromptTemplate(
    input_variables=["document"],
    template=(
        "Você é um assistente que produz resumos fiéis e concisos.\n"
        "Resuma o seguinte documento em até 5 bullet points, sem inventar informações.\n\n"
        "Documento:\n{document}"
    ),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

chain = prompt | llm
summary = chain.invoke({"document": text})
print(summary.content)
```

Exemplo 2: Sumarização com map_reduce (documento longo)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnableLambda

long_text = """Seu texto longo... (muito acima da janela do modelo)"""

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = splitter.split_text(long_text)

# Prompt para map (resumo por chunk)
map_prompt = PromptTemplate(
    input_variables=["chunk"],
    template=(
        "Resuma fielmente o trecho abaixo em 3-5 bullets, sem perder detalhes críticos.\n\n{chunk}"
    ),
)

# Prompt para reduce (combinar parciais)
reduce_prompt = PromptTemplate(
    input_variables=["partials"],
    template=(
        "A seguir estão resumos parciais de um documento maior.\n"
        "Combine-os em um resumo único, conciso e sem redundâncias, com até 7 bullets.\n\n{partials}"
    ),
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Fase map: processa cada chunk
def map_fn(chunk: str) -> str:
    return (map_prompt | llm).invoke({"chunk": chunk}).content

mapped = [map_fn(c) for c in chunks]

# Fase reduce: combina resumos
partials_joined = "\n\n".join(mapped)
final_summary = (reduce_prompt | llm).invoke({"partials": partials_joined}).content
print(final_summary)
```

Observações de custo e performance
- stuff: 1 chamada ao modelo (rápido e barato), mas limitado pela janela de contexto.
- map_reduce: ~N chamadas (N = número de chunks) + 1 chamada de redução. Custo e latência crescem com N. Você pode paralelizar a fase map para reduzir tempo.
- refine (alternativa): processa iterativamente os chunks, refinando um resumo acumulado. Custo geralmente entre stuff e map_reduce; pode manter melhor coerência sequencial.

Checklist de decisão rápida
- Documento cabe no contexto com folga? Use stuff.
- Documento longo exige cobertura ampla? Use map_reduce ou refine.
- Precisa de formato estruturado (JSON/bullets)? Especifique claramente no prompt.
- Precisa de citações/trechos? Instrua o modelo a incluir referências de seção/linha quando possível.

### Guia prático: quando usar stuff, map_reduce ou sem split

- **Sem split (documento único curto)**: quando o texto cabe confortavelmente na janela do modelo com sobra para instruções e formato de saída. Menor custo e latência; preserva coesão global. Exemplo: `2-chains-e-processamento/8-sumarizacao-sem-split.py`.

- **Stuff + split (documento moderado, cabe após dividir)**: quando o documento é médio e você pode concatenar os chunks no prompt final sem exceder o contexto. Útil se quiser uma única chamada final mantendo boa cobertura. Custo baixo a moderado. Exemplo: `2-chains-e-processamento/6-sumarizacao-com-stuff.py` com `RecursiveCharacterTextSplitter`.

- **Map_reduce (documento longo/grande, acima do contexto)**: quando o material excede claramente a janela do modelo. Cada chunk é resumido (map) e os resumos são combinados (reduce), garantindo cobertura ampla. Custo/latência maiores, porém mais escalável e robusto. Exemplo: `2-chains-e-processamento/7-sumarizacao-com-mapreduce.py`.

Justificativas rápidas:
- **Precisão/coerência**: sem split tende a manter melhor contexto global em textos curtos; em textos muito longos, map_reduce mitiga perda de contexto por chunk.
- **Custo/latência**: sem split < stuff + split < map_reduce (tipicamente).
- **Escala**: map_reduce é a opção segura para volumes longos; stuff funciona bem até o limite efetivo da janela; sem split é ideal apenas para textos curtos.

## Diferenças entre abordagens de implementação de Map-Reduce

### Implementação Tradicional vs Manual

#### Arquivo 7: `7-sumarizacao-com-mapreduce.py` (Abordagem Tradicional)

**Características:**
- Usa a função **`load_summarize_chain()`** do LangChain
- Implementação **pré-construída** e **simplificada**
- Apenas 3 linhas de código para criar o pipeline:
```python
chain_summarize = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
response = chain_summarize.invoke({"input_documents": parts})
```

**Vantagens:**
- ✅ **Rápido** para implementar
- ✅ **Menos código** para manter
- ✅ **Simplicidade** na configuração
- ✅ **Logs automáticos** com `verbose=True`

**Desvantagens:**
- ❌ **Menos controle** sobre os prompts (mas é possível personalizar)
- ❌ **Configuração limitada** (apenas `chain_type="map_reduce"`)
- ❌ **Prompts padrão** do LangChain (mas customizáveis)
- ❌ **Menos visibilidade** do processo

**Nota importante sobre personalização:**
O `load_summarize_chain` **permite personalizar os prompts** através dos parâmetros `map_prompt` e `combine_prompt`:

```python
from langchain.prompts import PromptTemplate

# Prompt personalizado para a fase Map
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="Resuma o seguinte texto em 3 palavras: {text}"
)

# Prompt personalizado para a fase Reduce
combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="Combine os seguintes resumos em um resumo final: {text}"
)

# Usando prompts personalizados
chain_summarize = load_summarize_chain(
    llm, 
    chain_type="map_reduce", 
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    verbose=True
)
```

Isso significa que a abordagem tradicional **não é tão limitada** quanto parece - você pode ter controle sobre os prompts mantendo a simplicidade da implementação.

#### Arquivo 9: `9-sumarizacao-personalizado.py` (Abordagem Manual)

**Características:**
- Implementação **manual** usando **LangChain Expression Language (LCEL)**
- Controle **total** sobre cada etapa do processo
- Pipeline construído passo a passo com componentes customizáveis

**Estrutura do Pipeline:**
```python
# Pipeline modular e explícito
map_chain = map_prompt | llm | StrOutputParser()
map_stage = prepare_map_input | map_chain.map()
reduce_chain = reduce_prompt | llm | StrOutputParser()
pipeline = map_stage | prepare_reduce_input | reduce_chain
```

**Vantagens:**
- ✅ **Controle total** sobre cada etapa
- ✅ **Customização completa** dos prompts:
  - Map: `"Summarize the following text in 5 words:\n{text}"`
  - Reduce: `"Combine the following summaries into a single concise summary:\n{text}"`
- ✅ **Visibilidade** de todo o processo
- ✅ **Flexibilidade** para modificações
- ✅ **Debugging explícito** de cada chunk

**Desvantagens:**
- ❌ **Mais código** para manter
- ❌ **Maior complexidade**
- ❌ **Mais tempo** para implementar

### Comparação Detalhada

#### 1. Abordagem de Implementação

**Tradicional (Arquivo 7):**
```python
# Implementação simples
chain_summarize = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
response = chain_summarize.invoke({"input_documents": parts})
```

**Manual (Arquivo 9):**
```python
# Implementação modular
map_chain = map_prompt | llm | StrOutputParser()
map_stage = prepare_map_input | map_chain.map()
reduce_chain = reduce_prompt | llm | StrOutputParser()
pipeline = map_stage | prepare_reduce_input | reduce_chain
response = pipeline.invoke(parts)
```

#### 2. Controle e Customização

**Tradicional:**
- **Menos controle** sobre os prompts
- Usa prompts **padrão** do LangChain
- Configuração **limitada**

**Manual:**
- **Controle total** sobre os prompts
- **Customização completa** de cada etapa
- **Visibilidade** de cada processo

#### 3. Configurações de Chunking

**Arquivo 7:**
- `chunk_size=250, chunk_overlap=70`

**Arquivo 9:**
- `chunk_size=300, chunk_overlap=50`

#### 4. Visibilidade e Debugging

**Tradicional:**
- Usa `verbose=True` para logs automáticos
- Menos controle sobre o que é exibido

**Manual:**
- **Controle manual** sobre a exibição
- Função `prepare_documents()` com prints customizados
- **Debugging explícito** de cada chunk

### Casos de Uso

#### Use a Abordagem Tradicional quando:
- Precisa de uma solução **rápida** e **simples**
- Não precisa de customizações específicas
- Quer usar os prompts **padrão** do LangChain
- Desenvolvimento **prototipado** ou **MVP**

#### Use a Abordagem Manual quando:
- Precisa de **controle total** sobre o processo
- Quer **prompts específicos** (como "5 palavras")
- Precisa de **debugging** detalhado
- Quer **entender** como o Map-Reduce funciona internamente
- Desenvolvimento **produção** com requisitos específicos

### Resumo das Diferenças

| Aspecto | Tradicional (Arquivo 7) | Manual (Arquivo 9) |
|---------|-------------------------|-------------------|
| **Implementação** | Pré-construída | Manual com LCEL |
| **Código** | 3 linhas | ~50 linhas |
| **Controle** | Limitado (mas com personalização) | Total |
| **Customização** | Básica a Intermediária | Completa |
| **Prompts** | Padrão (mas customizáveis) | Personalizados |
| **Debugging** | Automático | Manual |
| **Visibilidade** | Limitada | Completa |
| **Flexibilidade** | Média | Alta |
| **Complexidade** | Baixa | Alta |
| **Tempo de desenvolvimento** | Rápido | Lento |
| **Manutenção** | Fácil | Complexa |

### Níveis de Personalização

#### 1. Abordagem Tradicional Básica
```python
# Sem personalização - usa prompts padrão
chain_summarize = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
```

#### 2. Abordagem Tradicional com Prompts Personalizados
```python
# Com personalização de prompts - melhor dos dois mundos
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="Resuma em 5 palavras: {text}"
)
combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="Combine em resumo final: {text}"
)

chain_summarize = load_summarize_chain(
    llm, 
    chain_type="map_reduce", 
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    verbose=True
)
```

#### 3. Abordagem Manual Completa
```python
# Controle total com LCEL - máxima flexibilidade
map_chain = map_prompt | llm | StrOutputParser()
map_stage = prepare_map_input | map_chain.map()
reduce_chain = reduce_prompt | llm | StrOutputParser()
pipeline = map_stage | prepare_reduce_input | reduce_chain
```

### Recomendação

- **Para aprendizado e prototipagem**: Use a abordagem **tradicional básica** (Arquivo 7)
- **Para produção com prompts específicos**: Use a abordagem **tradicional com personalização** (Arquivo 7 + prompts customizados)
- **Para controle total e debugging**: Use a abordagem **manual** (Arquivo 9)
- **Para entender o funcionamento interno**: Implemente a versão **manual** primeiro

## Agentes de IA

### Conceito Fundamental
- **Definição**: software que utiliza LLMs como núcleo para tomar decisões autônomas e executar ações
- **Autonomia**: capacidade de decidir qual ferramenta usar baseado no contexto
- **Flexibilidade**: pode combinar múltiplas ferramentas para resolver problemas complexos
- **Iterativo**: pode executar múltiplas ações em sequência até atingir o objetivo

### Arquitetura de Agentes

#### Componentes Principais
- **LLM Core**: modelo de linguagem que toma as decisões
- **Tools/Tools**: conjunto de ferramentas disponíveis
- **Memory**: armazenamento de contexto e histórico
- **Executor**: motor que executa as ações decididas
- **Parser**: interpreta as respostas do LLM

#### Fluxo de Execução
1. **Input**: recebe pergunta ou tarefa do usuário
2. **Reasoning**: LLM analisa e decide próxima ação
3. **Tool Selection**: escolhe ferramenta apropriada
4. **Execution**: executa a ação com a ferramenta
5. **Observation**: analisa resultado da ação
6. **Iteration**: repete até completar a tarefa

### Tipos de Ferramentas (Tools)

#### 1. Ferramentas Computacionais
- **Funções Python**: cálculos, processamento de dados
- **APIs REST**: integração com serviços externos
- **Bibliotecas**: pandas, numpy, scikit-learn
- **Scripts**: automação de tarefas

#### 2. Ferramentas de Dados
- **Bancos de dados**: consultas SQL, NoSQL
- **Arquivos**: leitura/escrita de documentos
- **APIs de dados**: serviços de terceiros
- **Vector stores**: busca semântica

#### 3. Ferramentas de Comunicação
- **Email**: envio e recebimento
- **Slack/Discord**: integração com chat
- **Webhooks**: notificações
- **SMS**: mensagens de texto

#### 4. Ferramentas de Busca
- **Web search**: busca na internet
- **Wikipedia**: consulta de informações
- **Documentação**: busca em docs técnicos
- **Knowledge bases**: bases de conhecimento

### Padrão ReAct (Reasoning + Acting)

#### Ciclo ReAct
1. **Thought**: agente pensa sobre a situação atual
2. **Action**: decide qual ação tomar
3. **Observation**: observa o resultado da ação
4. **Final Answer**: fornece resposta final quando objetivo é atingido

#### Exemplo de Execução
```
Thought: Preciso buscar informações sobre o clima atual
Action: usar ferramenta de busca na web
Observation: temperatura é 25°C, céu nublado
Thought: Agora posso responder ao usuário
Final Answer: O clima atual é 25°C com céu nublado
```

### Tipos de Agentes no LangChain

#### 1. Zero-Shot ReAct Agent
- **Características**: não tem memória de interações anteriores
- **Uso**: tarefas simples e independentes
- **Vantagem**: simplicidade e rapidez
- **Limitação**: não aprende com histórico

#### 2. Conversational ReAct Agent
- **Características**: mantém memória da conversa
- **Uso**: interações contínuas com usuário
- **Vantagem**: contexto conversacional
- **Aplicação**: chatbots inteligentes

#### 3. Plan-and-Execute Agent
- **Características**: cria plano antes de executar
- **Uso**: tarefas complexas com múltiplos passos
- **Vantagem**: melhor organização e eficiência
- **Aplicação**: automação de workflows

#### 4. Self-Ask-with-Search Agent
- **Características**: faz perguntas para si mesmo
- **Uso**: pesquisa e análise de informações
- **Vantagem**: exploração sistemática
- **Aplicação**: assistentes de pesquisa

### Implementação Prática

#### Configuração Básica
```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Definir ferramentas
tools = [
    Tool(name="Calculator", func=calculator_func),
    Tool(name="WebSearch", func=web_search_func)
]

# Inicializar agente
agent = initialize_agent(
    tools=tools,
    llm=OpenAI(),
    agent_type="zero-shot-react-description"
)
```

#### Customização de Ferramentas
- **Descrições claras**: ferramentas devem ter descrições precisas
- **Parâmetros**: definir inputs necessários
- **Validação**: verificar inputs antes da execução
- **Error handling**: tratar erros graciosamente

### Considerações de Design

#### Escolha de Ferramentas
- **Relevância**: ferramentas devem ser úteis para o domínio
- **Confiabilidade**: ferramentas devem ser estáveis
- **Performance**: considerar latência e custos
- **Segurança**: validar inputs e outputs

#### Limitações e Desafios
- **Hallucination**: agente pode "inventar" ferramentas
- **Loops infinitos**: agente pode ficar preso em ciclos
- **Custos**: múltiplas chamadas ao LLM aumentam custos
- **Complexidade**: debugging pode ser difícil

#### Boas Práticas
- **Tool descriptions**: descrições claras e específicas
- **Error handling**: tratamento robusto de erros
- **Logging**: registrar ações para debugging
- **Testing**: testar com cenários diversos
- **Monitoring**: acompanhar performance e custos

### Casos de Uso Comuns

#### 1. Assistente de Desenvolvimento
- **Ferramentas**: GitHub API, compiladores, testes
- **Aplicação**: automação de desenvolvimento
- **Exemplo**: criar PRs, executar testes, deploy

#### 2. Assistente de Pesquisa
- **Ferramentas**: web search, databases, APIs
- **Aplicação**: coleta e análise de informações
- **Exemplo**: relatórios de mercado, análise de dados

#### 3. Assistente de Suporte
- **Ferramentas**: knowledge base, ticketing, chat
- **Aplicação**: atendimento ao cliente
- **Exemplo**: resolução de problemas, escalação

#### 4. Assistente de Automação
- **Ferramentas**: APIs, scripts, schedulers
- **Aplicação**: automação de processos
- **Exemplo**: backup, monitoramento, relatórios

## Memória em LLMs

### Conceito Fundamental
- **LLMs são stateless**: não mantêm informações entre requisições
- **Cada interação é independente**: o modelo não "lembra" de conversas anteriores
- **Necessidade de contexto**: toda informação relevante deve ser enviada a cada requisição
- **Memória interna do modelo**: limitada ao contexto da requisição atual (context window)

### Tipos de Memória

#### 1. Memória de Curto Prazo (Session Memory)
- **Duração**: durante uma única sessão/conversa
- **Armazenamento**:
    - Variáveis em memória do servidor
    - Cache temporário (Redis, Memcached)
    - Estado da aplicação
- **Uso**: manter contexto durante uma conversa ativa
- **Limitação**: perdida quando a sessão termina

#### 2. Memória de Longo Prazo (Persistent Memory)
- **Duração**: persistente entre sessões
- **Armazenamento**:
    - Bancos de dados relacionais (PostgreSQL, MySQL)
    - Bancos NoSQL (MongoDB, DynamoDB)
    - Vector databases (Pinecone, Weaviate)
- **Funcionalidades**:
    - Restaurar conversas anteriores
    - Continuar de onde parou
    - Buscar contexto histórico relevante
    - Manter perfil do usuário

### Estratégias de Gerenciamento de Memória

#### 1. Sliding Window
- **Conceito**: janela deslizante de mensagens/tokens
- **Funcionamento**: mantém apenas as N mensagens mais recentes
- **Vantagem**: controle preciso do uso de tokens
- **Desvantagem**: pode perder contexto importante

#### 2. Sumarização
- **Objetivo**: reduzir volume de histórico mantendo informações essenciais
- **Técnicas**:
    - Resumir mensagens antigas
    - Extrair pontos-chave
    - Manter apenas informações relevantes
- **Benefício**: economia de tokens e custos

#### 3. Memória Seletiva
- **Conceito**: armazenar apenas informações relevantes
- **Critérios de seleção**:
    - Relevância para o contexto atual
    - Importância para o usuário
    - Frequência de referência
- **Implementação**: algoritmos de scoring e filtragem

### Implementação no LangChain

#### Componentes de Memória
- **ConversationBufferMemory**: armazena todo o histórico
- **ConversationBufferWindowMemory**: sliding window
- **ConversationSummaryMemory**: sumarização automática
- **ConversationTokenBufferMemory**: baseado em tokens
- **VectorStoreRetrieverMemory**: busca semântica

#### Integração com Chains
- Memória é injetada automaticamente nos prompts
- Contexto é adicionado antes das mensagens do usuário
- Suporte a múltiplos tipos de memória simultaneamente

### Considerações Práticas

#### Limitações de Context Window
- **Tokens limitados**: cada modelo tem um limite máximo
- **Custo crescente**: mais tokens = maior custo
- **Performance**: contextos muito grandes podem degradar qualidade

#### Estratégias de Otimização
- **Truncamento inteligente**: remover partes menos relevantes
- **Compressão**: técnicas de compactação de texto
- **Hierarquia**: estruturar informações por importância
- **Cache**: reutilizar processamentos anteriores

#### Segurança e Privacidade
- **Dados sensíveis**: não armazenar informações confidenciais
- **Criptografia**: proteger dados em trânsito e repouso
- **Retenção**: políticas de limpeza de dados
- **Compliance**: seguir regulamentações (LGPD, GDPR)