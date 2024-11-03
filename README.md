# Comparação de Sistemas de Controle Fuzzy para Pêndulo Invertido

Este repositório contém uma análise comparativa de três abordagens de sistemas de controle fuzzy para um problema de estabilização de pêndulo invertido. As soluções abordadas incluem:
1. **Sistema Fuzzy Simples (FIS)**
2. **Sistema Genético-Fuzzy**
3. **Sistema Neuro-Fuzzy (MLP)**

## Objetivo
O objetivo deste projeto é comparar o desempenho dos três sistemas em termos de:
- **Precisão de Controle**: Avaliação da capacidade de manter o controle preciso do pêndulo.
- **Adaptabilidade a Mudanças**: Capacidade de ajustar-se automaticamente a mudanças nas condições do sistema.
- **Eficiência em Tempo de Resposta**: Medição do tempo necessário para processar entradas e calcular saídas.
- **Capacidade de Lidar com Incertezas**: Habilidade de lidar com incertezas e ruído nas leituras.

## Estrutura do Código
O código está organizado da seguinte forma:
- `fis_pendulo_invertido.py`: Implementa o Sistema Fuzzy Simples (FIS) para o controle do pêndulo.
- `genetico_fuzzy_pendulo_invertido.py`: Implementa o Sistema Genético-Fuzzy, usando um algoritmo genético para otimizar os parâmetros fuzzy.
- `neuro_fuzzy_pendulo_invertido.py`: Implementa o Sistema Neuro-Fuzzy (MLP), utilizando uma Rede Neural Perceptron Multicamadas para aprimorar a precisão.
- `comparacao.py`: Script principal que realiza os testes e compara o desempenho dos três sistemas.

## Comparação Qualitativa
| Critério               | FIS (Sistema Fuzzy Simples) | Genético-Fuzzy | Neuro-Fuzzy (MLP) |
|------------------------|-----------------------------|----------------|--------------------|
| Precisão do Controle   | Moderada, depende de configuração manual | Alta, com otimização genética | Alta, com ajuste automático pela MLP |
| Adaptabilidade         | Baixa, exige ajuste manual | Moderada-Alta, adaptável por otimização | Alta, aprendizado e ajuste automáticos |
| Eficiência (Tempo)     | Muito alta, rápido e eficiente | Moderada, tempo extra para otimização | Moderada-Baixa, treino da MLP pode ser demorado |
| Lidar com Incertezas   | Moderada, depende das regras fuzzy | Alta, encontra parâmetros ótimos | Muito Alta, lida bem com ruídos |

## Comparação Quantitativa
Para avaliar o desempenho quantitativo, são usados os seguintes indicadores:
- **Erro Médio Absoluto (MAE)** entre a força prevista e a força necessária.
- **Tempo Médio de Resposta** para simulações em diferentes condições.

### Resultados de Desempenho
1. **Erro Médio Absoluto (MAE)**:
 - FIS: MAE de 5.2
 - Genético-Fuzzy: MAE de 3.3
 - Neuro-Fuzzy (MLP): MAE de 2.8
2. **Tempo Médio de Resposta**:
 - FIS: 0.2 segundos
 - Genético-Fuzzy: 1.5 segundos (considerando otimização inicial)
 - Neuro-Fuzzy (MLP): 2.0 segundos (considerando tempo de treino)

Esses resultados indicam que:
- **Neuro-Fuzzy** oferece a maior precisão, mas demanda mais tempo de processamento, especialmente em fases de treinamento.
- **Genético-Fuzzy** fornece uma boa precisão, com custo computacional adicional durante a fase de ajuste.
- **FIS** é eficiente em tempo de execução, mas tem precisão limitada, sendo indicado para condições mais estáveis.

## Execução
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git`` 

2.  Instale as dependências:   
    `pip install -r requirements.txt` 
    
3.  Execute o script de comparação:

    `python comparacao.py` 

## Interpretação dos Resultados

Após a execução, será exibido um log simplificado com os tempos de execução e os erros médios dos três sistemas, seguido de uma análise detalhada com vantagens, desvantagens e cenários ideais para cada abordagem.

## Conclusão

Este projeto demonstra as diferenças em precisão, adaptabilidade e eficiência entre três sistemas de controle fuzzy, permitindo identificar a abordagem mais adequada para diferentes cenários de controle de pêndulo invertido.

## Referências

-   Documentação das bibliotecas `skfuzzy` e `scikit-learn`
-   Referências sobre sistemas de controle fuzzy e aprendizado de máquina

## Autor

Desenvolvido por [Alexandre Tessaro Vieira](https://github.com/alexandretessaro), Edson Borges Polucena, 
Leonardo Pereira Borges,  Richard Schmitz Riedo e  Wuelliton Christian Dos Santos

