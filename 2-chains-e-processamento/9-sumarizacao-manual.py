"""
Exemplo de Sumarização Manual com Map-Reduce
==================================================

Este script demonstra como implementar um pipeline de sumarização manual
usando o padrão Map-Reduce com LangChain Expression Language (LCEL).

Fluxo do Processamento:
1. Carregar e dividir texto longo em chunks menores
2. Aplicar sumarização em cada chunk (Map)
3. Combinar as sumarizações em um resumo final (Reduce)
"""

# ============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÃO INICIAL
# ============================================================================

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# Carrega variáveis de ambiente (incluindo OPENAI_API_KEY)
load_dotenv()

# ============================================================================
# 2. DADOS DE ENTRADA - TEXTO LONGO PARA SUMARIZAR
# ============================================================================

long_text = """
As profundezas do oceano são, ao mesmo tempo, um mistério e uma fronteira. Embora cubram mais de 70% da superfície do planeta, menos de 5% dos oceanos foram explorados em detalhes. O que se esconde abaixo da superfície azul é um mundo de escuridão, pressão extrema e criaturas que parecem pertencer a outro planeta — um reino silencioso e quase intocado, onde a vida encontrou maneiras surpreendentes de existir.

Logo abaixo da zona iluminada, onde a luz do sol ainda penetra e permite a fotossíntese, encontra-se a chamada **zona crepuscular**, uma região onde a luminosidade começa a desaparecer. Aqui, as cores se tornam difusas, o azul domina o cenário e a maioria das criaturas vive em constante penumbra. Muitos animais dessa faixa desenvolveram olhos enormes e órgãos bioluminescentes, capazes de gerar luz própria — uma adaptação essencial para caçar, atrair parceiros ou confundir predadores.

Mais abaixo, a escuridão é absoluta. A **zona abissal** começa a cerca de 4.000 metros de profundidade e se estende até aproximadamente 6.000 metros. A pressão nesse ambiente é esmagadora — milhares de vezes maior que a atmosférica na superfície. Nenhuma luz solar chega até lá. Mesmo assim, a vida prospera: peixes translúcidos, crustáceos de aparência alienígena e microrganismos que sobrevivem sem oxigênio compõem ecossistemas inteiros sustentados por fontes químicas em vez de energia solar.

Nos locais onde o fundo do mar se rasga e o calor do interior da Terra escapa, surgem as **chaminés hidrotermais** — verdadeiros oásis em meio à escuridão. Desses respiradouros jorra água superaquecida, rica em minerais e gases como sulfeto de hidrogênio. É nesse ambiente extremo que vivem bactérias quimiossintéticas, capazes de transformar substâncias químicas em energia. Elas formam a base de uma cadeia alimentar completamente independente da luz solar, sustentando vermes gigantes, camarões cegos e uma variedade impressionante de formas de vida únicas.

Ainda mais abaixo, em fossas oceânicas como a **Fossa das Marianas**, a mais profunda conhecida, o fundo do mar desce a quase 11.000 metros. Pouquíssimos veículos submersíveis já alcançaram tamanha profundidade. O ambiente é inóspito, frio, e a pressão é tão colossal que esmagaria qualquer submarino convencional em segundos. Ainda assim, até ali, organismos foram encontrados — pequenos crustáceos, bactérias e até peixes adaptados a condições que desafiam qualquer noção de sobrevivência.

O oceano profundo é também uma cápsula do tempo. Nele se acumulam sedimentos, fósseis e vestígios da história geológica da Terra. As correntes abissais moldam o clima global, regulam o ciclo do carbono e influenciam a atmosfera de formas que a ciência ainda tenta compreender. O desconhecido que habita as profundezas não é apenas biológico, mas também físico e químico — cada nova expedição revela espécies inéditas, compostos com potencial medicinal e fenômenos naturais que desafiam a imaginação humana.

Ao mesmo tempo, as profundezas enfrentam ameaças crescentes. O lixo plástico, as mudanças climáticas e a exploração mineral começam a atingir até mesmo regiões que, por milhões de anos, permaneceram intocadas. Microplásticos já foram encontrados em organismos a mais de 10.000 metros de profundidade. Isso revela uma triste ironia: o ser humano conseguiu deixar sua marca até no lugar mais remoto do planeta.

Explorar o oceano profundo é, em essência, explorar o desconhecido. É confrontar o que ainda não compreendemos sobre a vida, sobre a Terra e, de certo modo, sobre nós mesmos. Porque nas águas frias e escuras das profundezas marinhas está guardado um lembrete poderoso: a maior parte do nosso mundo ainda está por descobrir.
"""

# ============================================================================
# 3. DIVISÃO DO TEXTO EM CHUNKS (SPLITTING)
# ============================================================================

# Configuração do divisor de texto:
# - chunk_size=300: cada pedaço terá no máximo 300 caracteres
# - chunk_overlap=50: sobreposição de 50 caracteres entre chunks para manter contexto
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# Divide o texto longo em documentos menores
parts = splitter.create_documents([long_text])

# Opcional: visualizar os chunks gerados
# for part in parts:
#     print(f"Chunk {parts.index(part) + 1}:")
#     print(part.page_content)
#     print("-" * 50)

# ============================================================================
# 4. CONFIGURAÇÃO DO MODELO DE LINGUAGEM
# ============================================================================

# Inicializa o modelo OpenAI com configurações específicas:
# - model="gpt-4o-mini": modelo mais rápido e econômico
# - temperature=0: saída determinística (sem criatividade)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================================
# 5. PIPELINE MAP (SUMARIZAÇÃO DE CADA CHUNK)
# ============================================================================

# Template de prompt para sumarizar cada chunk em 5 palavras
map_prompt = PromptTemplate.from_template("Summarize the following text in 5 words:\n{text}")

# Cadeia LCEL para o processo Map:
# 1. Aplica o template de prompt
# 2. Envia para o modelo de linguagem
# 3. Converte a saída para string
map_chain = map_prompt | llm | StrOutputParser()

# Função para preparar os dados para o Map:
# Converte documentos em formato adequado para o prompt

def prepare_documents(docs):
    result = []
    for doc in docs:
        print(doc.page_content)
        print("-" *60)
        item = {"text": doc.page_content}
        result.append(item)
    return result

prepare_map_input = RunnableLambda(prepare_documents)

# prepare_map_input = RunnableLambda(lambda docs: [{"text": d.page_content} for d in docs])

# Estágio Map: aplica a sumarização em cada chunk
map_stage = prepare_map_input | map_chain.map()

# ============================================================================
# 6. PIPELINE REDUCE (COMBINAÇÃO DAS SUMARIZAÇÕES)
# ============================================================================

# Template de prompt para combinar as sumarizações em um resumo final
reduce_prompt = PromptTemplate.from_template("Combine the following summaries into a single concise summary:\n{text}")

# Cadeia LCEL para o processo Reduce:
# 1. Aplica o template de prompt
# 2. Envia para o modelo de linguagem
# 3. Converte a saída para string
reduce_chain = reduce_prompt | llm | StrOutputParser()

# Função para preparar os dados para o Reduce:
# Junta todas as sumarizações em uma única string
prepare_reduce_input = RunnableLambda(lambda summaries: {"text": "\n".join(summaries)})

# ============================================================================
# 7. PIPELINE COMPLETO (MAP + REDUCE)
# ============================================================================

# Combina os estágios Map e Reduce em um pipeline completo:
# 1. Map: sumariza cada chunk individualmente
# 2. Prepare: prepara os dados para o Reduce
# 3. Reduce: combina as sumarizações em um resumo final
pipeline = map_stage | prepare_reduce_input | reduce_chain

# ============================================================================
# 8. EXECUÇÃO E RESULTADO
# ============================================================================

# Executa o pipeline completo nos chunks do texto
response = pipeline.invoke(parts)

# Exibe o resultado final
print("=" * 60)
print("RESUMO FINAL DO TEXTO:")
print("=" * 60)
print(response)
print("=" * 60)
