# Importações necessárias para o gerenciamento de memória de conversas
from langchain_openai import ChatOpenAI  # Cliente para interagir com a API da OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Templates de prompt e placeholder para histórico
from langchain_core.chat_history import InMemoryChatMessageHistory  # Armazenamento de histórico em memória
from langchain_core.runnables.history import RunnableWithMessageHistory  # Runnable que mantém histórico de mensagens
from langchain_core.messages import trim_messages  # Função para trimar o histórico de mensagens
from langchain_core.runnables import RunnableLambda  # Runnable que executa uma função lambda
from dotenv import load_dotenv  # Para carregar variáveis de ambiente do arquivo .env

# Carrega as variáveis de ambiente (incluindo a chave da API da OpenAI)
load_dotenv()

# Criação do template de prompt para a conversa
# Este template define a estrutura das mensagens que serão enviadas para o modelo
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions."),  # Mensagem do sistema que define o comportamento do assistente
    MessagesPlaceholder(variable_name="history"),  # Placeholder onde será inserido o histórico de mensagens anteriores
    ("human", "{input}"),  # Template para a mensagem do usuário (o {input} será substituído pela entrada real)
])

# Inicialização do modelo de linguagem da OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)  # Usa o modelo GPT-4o-mini com temperatura 0.5 (criatividade moderada)

def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    trimmed_history = trim_messages(
        raw_history,
        token_counter=len,
        max_tokens=2,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    return {"input": payload["input"], "history": trimmed_history}

prepare = RunnableLambda(prepare_inputs)

chain = prepare | prompt | llm

# Dicionário para armazenar o histórico de conversas por sessão
# Cada chave é um session_id e o valor é um objeto InMemoryChatMessageHistory
session_store: dict[str, InMemoryChatMessageHistory] = {}

# Função para obter ou criar o histórico de uma sessão específica
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    # Se a sessão não existe no dicionário, cria uma nova instância de histórico
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    # Retorna o histórico da sessão (existente ou recém-criado)
    return session_store[session_id]

# Criação da cadeia conversacional com gerenciamento de histórico
# Esta é a peça principal que adiciona funcionalidade de memória à cadeia básica
conversational_chain = RunnableWithMessageHistory(
    chain,  # A cadeia básica (prompt + modelo) que será executada
    get_session_history,  # Função que retorna o histórico para uma sessão específica
    input_messages_key="input",  # Chave que identifica a entrada do usuário no dicionário de entrada
    history_messages_key="raw_history",  # Chave que identifica onde o histórico será inserido no prompt
)

# Configuração da sessão - define qual sessão será usada para armazenar o histórico
config = {"configurable": {"session_id": "1"}}

# Primeira interação: o usuário se apresenta
# O modelo não tem contexto anterior, então responde baseado apenas na mensagem atual
response1 = conversational_chain.invoke({"input": "Hello, my name is Igor."}, config=config)

# Segunda interação: o usuário pergunta sobre o próprio nome
# Agora o modelo tem acesso ao histórico da conversa e pode "lembrar" que o nome é Igor
response2 = conversational_chain.invoke({"input": "Tell me a joke. Don't mention my name."}, config=config)

# Terceira interação: o usuário pergunta sobre o próprio nome
# Agora o modelo tem acesso ao histórico da conversa e pode "lembrar" que o nome é Igor
response3 = conversational_chain.invoke({"input": "What is my name?"}, config=config)

# Exibição dos resultados
print(f"response1: {response1.content}")  # Resposta à primeira mensagem
print("-" * 100)
print(f"response2: {response2.content}")  # Resposta à segunda mensagem (deve "lembrar" o nome)
print("-" * 100)
print(f"response3: {response3.content}")  # Resposta à terceira mensagem (deve "lembrar" o nome)
print("-" * 100)
print(f"history: {get_session_history('1')}")  # Exibe todo o histórico armazenado da sessão