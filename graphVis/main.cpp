#include "hnswlib.h"
#include <iostream>

int main() {
    int dim = 2;              
    int max_elements = 10;
    int M = 4;                
    int ef_construction = 20;

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    std::cout << "inicio_dados_entrada" << std::endl;
 
    // Generate random data
    std::mt19937 rng;
    rng.seed(42);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
        std::cout << data[i] << " ";
        if ((i+1) % dim == 0)
            std::cout << "\n";
    }
    
    std::cout << "final_dados_entrada" << std::endl;

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }


    std::cout << "inicio_resultado_layer_0" << std::endl;

    int layer = 0;
    //std::cout << "Layer: " << layer << std::endl;
    for (int k = 0; k < max_elements ; k++) {
        int *datar = nullptr; // Inicialize com nullptr para evitar problemas de ponteiro nulo

        datar = (int *)alg_hnsw->get_linklist_at_level(k,layer);

        size_t size = alg_hnsw->getListCount((hnswlib::linklistsizeint*)datar);
        hnswlib::tableint *datal = (hnswlib::tableint *) (datar + 1);

        //if (datar != nullptr) {
        //    std::cout << "Elemento: "<< k << " size : " << alg_hnsw->getListCount((hnswlib::linklistsizeint*)datar) <<" links: ";
        //    for (size_t i = 0; i < size; i++) {
        //        std::cout << datal[i] << " ";
         //   }
        //    std::cout << std::endl;
        //} else {
        //    std::cout << "data é um ponteiro nulo." << std::endl;
        //}
        
        
        if (datar != nullptr) {
    	     std::cout << k << ": [";
    	     for (size_t i = 0; i < size; i++) {
        	std::cout << datal[i];
        
                if (i < size - 1) {
                     std::cout << ", ";
                }
             }
             std::cout << "]" << std::endl;
        } else {
             std::cout << "data é um ponteiro nulo." << std::endl;
          }
        }

    std::cout << "final_resultado_layer_0" << std::endl;
//    int layer1 = 1; // Ou outro valor apropriado

//    for (int k = 0; k < 1 ; k++) {
//        int *datar = nullptr; // Inicialize com nullptr para evitar problemas de ponteiro nulo

//        datar = (int *)alg_hnsw->get_linklist_at_level(14,layer1);

//        size_t size = alg_hnsw->getListCount((hnswlib::linklistsizeint*)datar);
//        hnswlib::tableint *datal = (hnswlib::tableint *) (datar + 1);

//        if (datar != nullptr) {
//            std::cout << "links no elemento ["<< k <<"]: ";
//            for (size_t i = 0; i < size; i++) {
//                std::cout << datar[i] << " ";
//            }
//            std::cout << std::endl;
//        } else {
//            std::cout << "data é um ponteiro nulo." << std::endl;
//        }
//    }

//    int layer2 = 2; // Ou outro valor apropriado

//    for (int k = 0; k < 1 ; k++) {
//        int *datar = nullptr; // Inicialize com nullptr para evitar problemas de ponteiro nulo

//        datar = (int *)alg_hnsw->get_linklist_at_level(k,layer2);

//        size_t size = alg_hnsw->getListCount((hnswlib::linklistsizeint*)datar);
//        hnswlib::tableint *datal = (hnswlib::tableint *) (datar + 1);

//        if (datar != nullptr) {
//            std::cout << "links no elemento ["<< k <<"]: ";
//            for (size_t i = 0; i < size; i++) {
//                std::cout << datar[i] << " ";
//            }
//            std::cout << std::endl;
//        } else {
//            std::cout << "data é um ponteiro nulo." << std::endl;
//        }
//    }



    // Suponha que você queira acessar o elemento com índice interno 1
    hnswlib::tableint internal_id = 1;

    // Acesse os dados do elemento
    char *element_data = alg_hnsw->getDataByInternalId(internal_id);

    // Verifique se o ponteiro não é nulo (para garantir que o elemento exista)
    if (element_data != nullptr) {
        // Agora você pode acessar os dados do elemento. A estrutura e o formato dos dados
        // dependem de como você os armazenou originalmente.
        // Suponha que seus dados sejam floats, você pode acessá-los da seguinte forma:
        float *float_data = reinterpret_cast<float*>(element_data);

        // Imprima os valores dos dados do elemento
        for (int i = 0; i < dim; i++) {
            std::cout << float_data[i] << " ";
        }
        std::cout << std::endl;
    } else {
        // O elemento com o índice interno especificado não existe.
        std::cout << "Elemento não encontrado." << std::endl;
    }

    delete[] data;
    delete alg_hnsw;
    return 0;
}
