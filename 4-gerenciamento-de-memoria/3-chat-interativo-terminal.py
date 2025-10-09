#!/usr/bin/env python3
"""
Chat Interativo no Terminal com Histórico em Memória
Baseado no exemplo de gerenciamento de memória do LangChain

Este script permite que você converse com o modelo GPT através do terminal,
mantendo o histórico da conversa em memória durante toda a sessão.
"""

# Importações necessárias para o chat interativo
from langchain_openai import ChatOpenAI  # Cliente para interagir com a API da OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Templates de prompt e placeholder para histórico
from langchain_core.chat_history import InMemoryChatMessageHistory  # Armazenamento de histórico em memória
from langchain_core.runnables import RunnableWithMessageHistory  # Runnable que mantém histórico de mensagens
from dotenv import load_dotenv  # Para carregar variáveis de ambiente do arquivo .env
import sys  # Para sair do programa
import os  # Para limpar a tela

# Carrega as variáveis de ambiente (incluindo a chave da API da OpenAI)
load_dotenv()

def limpar_tela():
    """Limpa a tela do terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def exibir_banner():
    """Exibe o banner de boas-vindas do chat"""
    print("=" * 60)
    print("🤖 CHAT INTERATIVO COM GPT - LANGCHAIN")
    print("=" * 60)
    print("Comandos especiais:")
    print("  /sair ou /quit - Sair do chat")
    print("  /limpar ou /clear - Limpar histórico da conversa")
    print("  /ajuda ou /help - Exibir esta ajuda")
    print("  /histórico ou /history - Ver histórico da conversa")
    print("=" * 60)
    print()

def configurar_chat():
    """
    Configura o chat com modelo, prompt e gerenciamento de histórico
    Retorna a cadeia conversacional configurada
    """
    
    # Criação do template de prompt para a conversa
    # Este template define a estrutura das mensagens que serão enviadas para o modelo
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions in Portuguese. Be friendly and conversational."),  # Mensagem do sistema que define o comportamento do assistente
        MessagesPlaceholder(variable_name="history"),  # Placeholder onde será inserido o histórico de mensagens anteriores
        ("user", "{input}"),  # Template para a mensagem do usuário (o {input} será substituído pela entrada real)
    ])

    # Inicialização do modelo de linguagem da OpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Usa o modelo GPT-4o-mini com temperatura 0.7 (criatividade moderada)

    # Criação da cadeia básica: prompt seguido do modelo de linguagem
    # O operador | (pipe) conecta o prompt ao modelo, criando um fluxo de processamento
    chain = prompt | llm

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
        history_messages_key="history",  # Chave que identifica onde o histórico será inserido no prompt
    )

    return conversational_chain, get_session_history

def exibir_historico(get_session_history_func, session_id="chat"):
    """
    Exibe o histórico completo da conversa
    """
    print("\n" + "=" * 50)
    print("📜 HISTÓRICO DA CONVERSA")
    print("=" * 50)
    
    history = get_session_history_func(session_id)
    messages = history.messages
    
    if not messages:
        print("Nenhuma mensagem no histórico ainda.")
    else:
        for i, message in enumerate(messages, 1):
            role = "👤 Usuário" if message.type == "human" else "🤖 Assistente"
            content = message.content
            print(f"{i}. {role}: {content}")
    
    print("=" * 50)
    print()

def processar_comando(comando, conversational_chain, get_session_history_func, config):
    """
    Processa comandos especiais do chat
    Retorna True se deve continuar o chat, False se deve sair
    """
    comando = comando.lower().strip()
    
    if comando in ['/sair', '/quit', '/exit']:
        print("👋 Até logo! Obrigado por usar o chat!")
        return False
    
    elif comando in ['/limpar', '/clear']:
        # Limpa o histórico da sessão atual
        session_id = config["configurable"]["session_id"]
        get_session_history_func(session_id).clear()
        print("🧹 Histórico da conversa foi limpo!")
        return True
    
    elif comando in ['/ajuda', '/help']:
        exibir_banner()
        return True
    
    elif comando in ['/histórico', '/history']:
        exibir_historico(get_session_history_func, config["configurable"]["session_id"])
        return True
    
    else:
        print("❓ Comando não reconhecido. Digite /ajuda para ver os comandos disponíveis.")
        return True

def main():
    """
    Função principal que executa o loop do chat interativo
    """
    try:
        # Limpa a tela e exibe o banner
        limpar_tela()
        exibir_banner()
        
        # Configura o chat
        print("🔧 Configurando o chat...")
        conversational_chain, get_session_history_func = configurar_chat()
        
        # Configuração da sessão - define qual sessão será usada para armazenar o histórico
        config = {"configurable": {"session_id": "chat_terminal"}}
        
        print("✅ Chat configurado com sucesso!")
        print("💬 Digite sua mensagem ou um comando especial:")
        print()
        
        # Loop principal do chat
        while True:
            try:
                # Solicita entrada do usuário
                user_input = input("👤 Você: ").strip()
                
                # Verifica se a entrada está vazia
                if not user_input:
                    print("⚠️  Por favor, digite uma mensagem ou comando.")
                    continue
                
                # Verifica se é um comando especial
                if user_input.startswith('/'):
                    if not processar_comando(user_input, conversational_chain, get_session_history_func, config):
                        break
                    continue
                
                # Processa a mensagem do usuário
                print("🤖 Assistente: ", end="", flush=True)
                
                # Envia a mensagem para o modelo e obtém a resposta
                response = conversational_chain.invoke({"input": user_input}, config=config)
                
                # Exibe a resposta do assistente
                print(response.content)
                print()
                
            except KeyboardInterrupt:
                # Captura Ctrl+C para sair graciosamente
                print("\n\n👋 Chat interrompido pelo usuário. Até logo!")
                break
                
            except Exception as e:
                # Captura outros erros e exibe mensagem amigável
                print(f"\n❌ Erro ao processar mensagem: {e}")
                print("🔄 Tente novamente ou digite /sair para sair.")
                print()
    
    except Exception as e:
        # Captura erros de configuração
        print(f"❌ Erro ao configurar o chat: {e}")
        print("🔍 Verifique se:")
        print("   - Sua chave da API da OpenAI está configurada no arquivo .env")
        print("   - Você tem conexão com a internet")
        print("   - As dependências estão instaladas corretamente")
        sys.exit(1)

if __name__ == "__main__":
    main()
