from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

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

documents = [Document(page_content=long_text)]

print(documents[0].page_content)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain_summarize = load_summarize_chain(llm, chain_type="stuff", verbose=True)

response = chain_summarize.invoke({"input_documents": long_text})

print(response["output_text"])