import matplotlib.pyplot as plt
import re
import subprocess

def compilar_codigo(caminho_arquivo_cpp, caminho_executavel):
    comando_compilacao = f"g++ -o {caminho_executavel} {caminho_arquivo_cpp}"

    # Executar o comando de compilação
    try:
        subprocess.check_output(comando_compilacao, shell=True)
        print("Compilação bem-sucedida.")
    except subprocess.CalledProcessError as e:
        print(f"Erro durante a compilação: {e}")

def alterar_valor_max_elements(caminho_arquivo, novo_valor):
    # Ler o conteúdo do arquivo
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()

    # Procurar a linha contendo "int max_elements"
    for i, linha in enumerate(linhas):
        if "int max_elements" in linha:
            # Preservar a indentação e substituir apenas o valor
            linhas[i] = linha.split('=')[0] + f'= {novo_valor};\n'

    # Escrever de volta no arquivo
    with open(caminho_arquivo, 'w') as arquivo:
        arquivo.writelines(linhas)



def extrair_parte_resultado(resultado_final):
    padrao = re.compile(r'inicio_resultado_layer_0(.*?)final_resultado_layer_0', re.DOTALL)
    correspondencia = padrao.search(resultado_final)

    if correspondencia:
        parte_extraida = correspondencia.group(1).strip()
        #print(parte_extraida)
        return parte_extraida
    else:
        print("Não foi possível encontrar a parte desejada.")
        return
        
def extrair_parte_resultado_dict(resultado_final):
    padrao = re.compile(r'(\d+): \[(.*?)\]', re.DOTALL)
    correspondencias = padrao.findall(resultado_final)

    if correspondencias:
        edges = {int(i): list(map(int, valores.split(', '))) for i, valores in correspondencias}
        #print(edges)
        return edges
    else:
        print("Não foi possível encontrar a parte desejada.")
        return {}

def processar_dados(dados):
    linhas = dados.strip().split('\n')
    for i, linha in enumerate(linhas):
        valores = linha.split()
        print(f"{i}: ({' '.join(valores)})")


def processar_dados_dict(dados):
    vertices = {}
    linhas = dados.strip().split('\n')
    
    for i, linha in enumerate(linhas):
        valores = linha.split()
        vertices[i] = tuple(map(float, valores))
    
    return vertices


def pegar_pontos_gerados(resultado_final):
    padrao = re.compile(r'inicio_dados_entrada(.*?)final_dados_entrada', re.DOTALL)
    correspondencia = padrao.search(resultado_final)
    if correspondencia:
        parte_dados = correspondencia.group(1)
        dict_data = processar_dados_dict(parte_dados)
        return dict_data
    else:
        print("Erro: inicio_dados_entrada ou final_dados_entrada não encontrado.")
        return {}


    
def plot_graph(vertices, edges):
    # Extrai as coordenadas dos vértices
    x = [vertex[0] for vertex in vertices.values()]
    y = [vertex[1] for vertex in vertices.values()]

    # Plota os vértices
    plt.scatter(x, y, color='blue')

    # Adiciona rótulos aos vértices
    for v, coord in vertices.items():
        plt.text(coord[0], coord[1], str(v), fontsize=12, ha='right', va='bottom')

    # Plota as arestas com setas
    for v, neighbors in edges.items():
        for neighbor in neighbors:
            dx = vertices[neighbor][0] - vertices[v][0]
            dy = vertices[neighbor][1] - vertices[v][1]

            # Adiciona a seta na direção da aresta
            plt.arrow(vertices[v][0], vertices[v][1], dx, dy, color='black', 
                    shape='full', lw=0, length_includes_head=True, head_width=0.022, head_length=0.043)

    # Configurações adicionais
    plt.title('Graph Visualization with Smaller Arrows')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

