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
    - Parseamento de saída para formatos específicos, como JSON
- Sumarização, utilizando map-reduce: 
    - Sumarização de textos longos em textos mais curtos.

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