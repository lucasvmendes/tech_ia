# 🧠 Classificação de Imagens Histopatológicas com Redes Neurais Convolucionais (BreaKHis)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![Accuracy](https://img.shields.io/badge/Acurácia-91.3%25-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 🗾 Resumo

Este projeto apresenta uma abordagem de **aprendizado profundo** para a **classificação binária de imagens histopatológicas de tumores mamários**, utilizando o banco de dados **BreaKHis**.  
O modelo, baseado em **Redes Neurais Convolucionais (CNNs)**, diferencia tecidos **benignos** e **malignos**, alcançando **acurácia média de 91,3%**.  
O estudo reforça o potencial da **Inteligência Artificial aplicada à patologia digital**, favorecendo diagnósticos assistidos e reprodutíveis.

---

## 🎯 Objetivo

Desenvolver e avaliar um pipeline de **visão computacional** para classificação automatizada de tumores mamários em imagens histológicas, com ênfase em:

- Processamento e normalização de imagens;  
- Construção e treinamento de uma CNN personalizada;  
- Avaliação quantitativa e comparação com estudos da literatura.  

---

## ⚙️ Metodologia

O projeto foi desenvolvido em **Python 3.10**, com bibliotecas:

`TensorFlow`, `Keras`, `NumPy`, `pandas`, `scikit-learn`, `matplotlib` e `seaborn`.

**Etapas principais:**
1. Pré-processamento e filtragem do dataset BreaKHis (`Folds.csv`);  
2. Divisão dos dados em treino/validação/teste;  
3. Construção da CNN com camadas convolucionais e de regularização;  
4. Treinamento com *EarlyStopping* e *ReduceLROnPlateau*;  
5. Avaliação com métricas quantitativas e geração de gráficos.

---
 
## 🗂️ Dataset

O conjunto de dados utilizado neste projeto é o **BreaKHis – Breast Cancer Histopathological Database**, disponibilizado pela **Universidade Federal do Paraná (UFPR)**.  
Disponível em: [https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

O BreaKHis contém **7.909 imagens histopatológicas** de tumores mamários, obtidas de **82 pacientes**, com ampliações de **40×, 100×, 200× e 400×**.  
As imagens estão divididas em duas classes principais:

- **Benignas:** Adenosis, Fibroadenoma, Tubular Adenoma, Phyllodes Tumor  
- **Malignas:** Carcinoma Ductal, Lobular, Mucinoso, Papilífero  

> **Uso:** Os dados foram utilizados exclusivamente para fins **acadêmicos e de pesquisa**, respeitando as condições de uso descritas no site oficial.

---

## 🧪 Protocolo de Avaliação

| Aspecto | Descrição |
|----------|------------|
| **Base de dados** | BreaKHis (Breast Cancer Histopathological Database) |
| **Magnificação** | Somente ≥ 200× |
| **Classes** | `benign` / `malignant` |
| **Divisão dos dados** | *Folds* predefinidos (`fold = 2`) |
| **Controle de viés** | Separacão por paciente para evitar sobreposição |
| **Validação cruzada** | 5-fold cross validation |
| **Métricas reportadas** | Média ± Desvio padrão (acurácia, precisão, recall, F1) |

---

## 🧩 Arquitetura da CNN

| Camada | Tipo / Parâmetros | Ativação |
|--------|-------------------|-----------|
| 1 | Conv2D (32, 3×3, same) | ReLU |
| 2 | MaxPooling2D (2×2) | — |
| 3 | Conv2D (64, 3×3, same) | ReLU |
| 4 | MaxPooling2D (2×2) | — |
| 5 | Dropout (0.3) | — |
| 6 | Flatten | — |
| 7 | Dense (128) | ReLU |
| 8 | Dropout (0.5) | — |
| 9 | Dense (1) | Sigmoid |

**Hiperparâmetros:**  
- Perda: `binary_crossentropy`  
- Otimizador: `Adam (lr = 0.001)`  
- Batch size: 32 | Épocas: 50 | Métrica: accuracy  
- Ambiente: Google Colab GPU (T4)  
- *Seed* fixa para reprodutibilidade  

---

## 📊 Resultados

| Métrica | Média | Desvio-padrão |
|----------|--------|---------------|
| **Acurácia** | 0.913 | ± 0.012 |
| **Precisão** | 0.897 | ± 0.018 |
| **Recall** | 0.924 | ± 0.015 |
| **F1-Score** | 0.910 | ± 0.014 |

> As curvas de aprendizado e a matriz de confusão indicam boa generalização e estabilidade entre folds.

**Comparativo com literatura:** desempenho compatível com Spanhol et al. (2016) e Araujo et al. (2017), confirmando eficácia da CNN aplicada.

---

## 🚀 Execução

```bash
# Clonar o repositório
git clone https://github.com/usuario/BreaKHis-CNN.git
cd BreaKHis-CNN

# Instalar dependências
pip install -r requirements.txt
```

### No Google Colab
1. Abra `VisaoComputacional-V10.ipynb`;  
2. Monte o Google Drive com `BreaKHis_v1` e `Folds.csv`;  
3. Ajuste o caminho base no início do notebook:  
   ```python
   path_prefix = "/content/drive/MyDrive/IA/Fiap-Alura/neto/"
   ```  
4. Execute as células em sequência.  
5. Resultados (curvas, matrizes) serão salvos em `/results`.

---

## 🧩 Conclusão

O modelo CNN alcançou **acurácia superior a 90%** na classificação binária de imagens histopatológicas de câncer de mama, demonstrando desempenho consistente e confiável.  
O projeto cumpre integralmente seus objetivos e encontra-se finalizado para fins **acadêmicos e de demonstração técnica**.

---

## 📚 Referências

1. Spanhol F. A. et al. (2016). *A Dataset for Breast Cancer Histopathological Image Classification.* IEEE Trans. Biomed. Eng. 63(7), 1455–1462.  
2. Araujo T. et al. (2017). *Classification of breast cancer histology images using Convolutional Neural Networks.* PLoS ONE 12(6): e0177544.  
3. Coudray N. et al. (2018). *Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning.* Nature Medicine 24(10): 1559–1567.  
4. LeCun Y., Bengio Y., & Hinton G. (2015). *Deep Learning.* Nature 521: 436–444.  
5. Chollet F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.* IEEE CVPR.  
6. TensorFlow Developers (2023). *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.* <https://www.tensorflow.org>  
7. Keras API Reference (2023). <https://keras.io>

---

## 👤 Autor

**Adalberto Ferreira de Albuquerque Neto**  
Pós-Graduação FIAP / Alura **Projeto desenvolvido no contexto de Tech Challenge 01**

