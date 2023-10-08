from functions import alterar_valor_max_elements, plot_graph, compilar_codigo, pegar_pontos_gerados, extrair_parte_resultado_dict
import subprocess

def main():

    # aletrar numero elementos 
    caminho_arquivo_main_cpp = './main.cpp'
    novo_valor_num_elements = 10
    alterar_valor_max_elements(caminho_arquivo_main_cpp, novo_valor_num_elements)

    #compilar codigo
    caminho_do_executavel = './execCpp'
    compilar_codigo(caminho_arquivo_main_cpp, caminho_do_executavel)

    # Executar o código C++ e pegar a saída
    resultado_cplusplus = subprocess.check_output(["./execCpp"])

    # Decodificar a saída para uma string Python
    resultado_final = resultado_cplusplus.decode("utf-8").strip()

    # pegar os pontos gerados 
    dict_data = pegar_pontos_gerados(resultado_final)
    #print(dict_data)

    # pegar os links 
    edges_dict = extrair_parte_resultado_dict(resultado_final)
    #print(edges_dict)

    plot_graph(dict_data,edges_dict)


if __name__ == "__main__":
    main()
