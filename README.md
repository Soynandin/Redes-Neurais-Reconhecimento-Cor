# Redes-Neurais-Reconhecimento-Cor
 Inteligência Artificial para Reconhecimento de Cores em imagens, criada com Streamlit e TensorFlow.

---

## Sistema de Reconhecimento de Cores em Imagens
Este projeto utiliza aprendizado de máquina e redes neurais convolucionais (CNNs) para realizar a **identificação de cores predominantes em imagens**. A solução foi desenvolvida com a biblioteca **TensorFlow**, aproveitando a capacidade de modelos de aprendizado profundo para análise de imagens. Integrando também **scikit-learn** para processamento de dados, o **NumPy** para manipulação de arrays, o **Pillow** para pré-processamento de imagens, o **OpenCV** para operações de visão computacional e **Streamlit** para criar uma interface interativa que facilita a interação do usuário com o modelo.

O sistema foi treinado para reconhecer cores específicas em imagens, como **vermelho**, **azul** e **verde**. A aplicação pode ser usada para prever qual a cor predominante em uma imagem carregada pelo usuário.

---

## Descrição do Sistema
O sistema é composto por várias partes:

1. **Pré-processamento de Imagens**: As imagens de treinamento são processadas, redimensionadas e normalizadas antes de serem alimentadas no modelo de rede neural.
   
2. **Treinamento do Modelo**: O modelo de rede neural convolucional (CNN) é treinado para classificar as cores predominantes nas imagens. Ele utiliza a arquitetura de redes neurais para aprender as características das cores nas imagens de entrada.
   
3. **Interface Interativa**: Usando o **Streamlit**, foi criada uma interface onde o usuário pode carregar uma imagem, e o sistema prediz qual é a cor predominante da imagem. O modelo treinado classifica a imagem como **vermelho**, **azul** ou **verde**.

4. **Predição em Tempo Real**: Após o treinamento, o modelo é salvo e pode ser utilizado para realizar previsões em novas imagens. As previsões podem ser visualizadas diretamente na interface interativa.

---

## Funcionalidades
- **Carregar uma Imagem**: O usuário pode carregar uma imagem (nos formatos `.jpg`, `.png`, `.jpeg`) através da interface.
- **Predição da Cor Predominante**: Após carregar a imagem, o sistema realizará a previsão da cor predominante (vermelho, azul ou verde).
- **Exibição da Imagem e Resultado**: A imagem carregada é exibida junto com a cor predominante prevista.

---

## Como Usar o Sistema

Este tutorial guiará você por todo o processo de clonagem do repositório, instalação das dependências, treinamento da IA pela primeira vez e utilização do sistema para prever as cores predominantes nas imagens.

### Instalação e Configuração

1. **Clone o repositório ou baixe o projeto:**:
   ```bash
   git clone https://github.com/Soynandin/Redes-Neurais-Reconhecimento-Cor.git
   cd redes_neurais_reconhecimento_de_cor

2. **Crie um ambiente virtual:**:
   ```bash
   python -m venv venv

3. **Ative o ambiente virtual:**:
   ```bash
    - No Windows: venv\Scripts\activate
    - No Linux/Mac: source venv/bin/activate

4. **Instale as dependências:**:
   ```bash
   pip install -r requirements.txt

5. **Organize os dados de treinamento:**:
- Certifique-se de que a pasta dados/treino contém subpastas (azul, verde, vermelho) com imagens representativas de cada classe.
- As imagens devem ser salvas nos formatos .jpg, .jpeg ou .png.

6. **Como Rodar o Sistema**:
 6.1. **Treinar o modelo**:
- Execute o script para treinar a rede neural: 

   ```bash
    python scripts/treinar_modelo.py

- O modelo treinado será salvo no diretório modelos como cnn_modelo_cor.h5.

 6.2. **Rodar a interface gráfica**:
- Após o treinamento, execute o aplicativo:

   ```bash
   streamlit run app.py

- Acesse o endereço fornecido pelo Streamlit (geralmente http://localhost:8501) no navegador.
- Carregue uma imagem para identificar a cor predominante.

---

## Observações

### Problemas Comuns
#### Erro ao carregar imagens:
- Verifique se as pastas dados/treino estão organizadas e contêm imagens válidas.

#### Erro no Streamlit:
- Certifique-se de que o ambiente virtual está ativado e as dependências foram instaladas corretamente.

#### Modelo não encontrado:
- Execute o treinamento antes de usar o aplicativo.


### Quantidade de Imagens no Conjunto de Treinamento
A quantidade de imagens no conjunto de treinamento tem um impacto significativo no desempenho do modelo de aprendizado de máquina, especialmente quando se trata de **redes neurais convolucionais (CNNs)**. Uma quantidade adequada de imagens é essencial para que o modelo possa aprender de forma eficaz e generalizar bem para novas imagens que não tenha visto durante o treinamento.

#### Efeitos de uma Quantidade Insuficiente de Imagens
- **Subajuste (Underfitting)**: Se a quantidade de imagens for muito pequena, o modelo pode não ser capaz de aprender características representativas das classes (cores) e, como resultado, terá um desempenho ruim tanto no treinamento quanto em novas imagens. Isso ocorre porque a rede neural não recebe informação suficiente para capturar a variabilidade dentro de cada classe.
  
- **Sobreajuste (Overfitting)**: Por outro lado, se a quantidade de imagens for muito pequena, mas o modelo for treinado por muitas iterações, o modelo pode memorizar as imagens do conjunto de treinamento, ou seja, ele pode aprender detalhes específicos demais que não são úteis para novas imagens. Nesse caso, o modelo pode ter um desempenho excelente no treinamento, mas falhar ao tentar classificar imagens desconhecidas.

#### Efeitos de uma Quantidade Adequada de Imagens
- **Generalização**: Quando o número de imagens é adequado, o modelo é capaz de aprender padrões gerais das cores predominantes nas imagens. Isso permite que ele classifique corretamente novas imagens que não estão no conjunto de treinamento.
  
- **Aumento da Precisão**: Quanto mais exemplos de imagens o modelo tiver, maior será a capacidade dele de identificar padrões robustos e, portanto, maior a precisão nas previsões. Isso é especialmente importante para imagens com variações sutis nas cores.

#### Recomendação para o Conjunto de Treinamento
Embora não haja um número exato de imagens que funcione para todos os tipos de problemas, uma boa prática é ter pelo menos **100 a 500 imagens por classe (cor)** para começar a ver bons resultados com redes neurais convolucionais. Esse número pode ser ajustado conforme o desempenho do modelo:

- Se o modelo não está generalizando bem, ou seja, se está cometendo erros frequentes em imagens de teste, considere aumentar a quantidade de imagens.
- Se o conjunto de dados for muito grande, você também pode experimentar técnicas de **aumento de dados (data augmentation)** para gerar variações artificiais das imagens, o que pode ajudar a melhorar a robustez do modelo sem a necessidade de coletar mais dados.

#### Aumento de Dados
Uma estratégia comum para melhorar a performance do modelo com conjuntos de dados limitados é o **aumento de dados**. Isso envolve aplicar transformações simples, como rotação, zoom, e variação de brilho nas imagens, criando novos exemplos de treinamento a partir de imagens já existentes. O uso de aumento de dados pode ajudar a simular uma maior diversidade de cenários e melhorar a capacidade do modelo de generalizar.

### Cores das Imagens e o Modelo de IA
As **cores predominantes** nas imagens são uma parte fundamental para a classificação e previsão feitas pelo modelo de aprendizado de máquina. Este sistema utiliza **redes neurais convolucionais (CNNs)** para aprender a identificar padrões de cores a partir das imagens que são fornecidas no treinamento.

#### Representação de Cores: RGB
As cores das imagens são geralmente representadas no modelo **RGB** (Red, Green, Blue), onde cada cor é uma combinação de três componentes: **vermelho (Red)**, **verde (Green)** e **azul (Blue)**. Cada uma dessas componentes tem valores numéricos entre 0 e 255, representando a intensidade da cor em cada canal:

- **Vermelho (R)**: Intensidade da cor vermelha.
- **Verde (G)**: Intensidade da cor verde.
- **Azul (B)**: Intensidade da cor azul.

Por exemplo, a cor **branca** seria representada por `(255, 255, 255)`, enquanto a cor **preta** seria `(0, 0, 0)` e o **vermelho** seria `(255, 0, 0)`.

#### Influência das Cores na IA
O modelo de IA depende fortemente da informação contida nas cores das imagens para aprender a identificar padrões. Algumas considerações importantes sobre como as cores influenciam a decisão do modelo:

- **Variedade nas Cores**: Imagens com uma diversidade de tonalidades e intensidades de cores podem fornecer mais informações ao modelo, permitindo que ele aprenda a distinguir melhor entre diferentes classes (por exemplo, entre as cores predominantes vermelho, verde e azul).
  
- **Normalização das Cores**: Durante o pré-processamento, as imagens são normalizadas para ter valores de pixel entre 0 e 1 (dividindo por 255), o que ajuda o modelo a aprender de forma mais eficiente. Isso permite que a rede neural perceba a diferença entre as cores de forma mais suave e com maior precisão.

- **Atenção a Cores Predominantes**: O modelo é treinado para identificar as cores predominantes nas imagens, ou seja, as cores que estão mais presentes e são mais "fortes" visualmente. Isso pode afetar diretamente a precisão da previsão do modelo, pois se uma cor não for predominante o suficiente, a IA pode ter dificuldades em identificá-la corretamente.

- **Ruído nas Imagens**: Imagens que contêm **muito ruído** (variações de cor não relevantes ou artefatos da câmera) podem confundir o modelo. É importante que as imagens sejam de boa qualidade, com cores bem definidas e sem distorções, para que o modelo possa aprender as características de cada classe de cor de forma mais eficiente.

#### Impacto no Desempenho do Modelo
O modelo de IA tenta associar as características de cada cor aos rótulos correspondentes durante o treinamento. Uma **maior diversidade de imagens** com uma gama variada de tonalidades e cores irá ajudar o modelo a aprender a reconhecer as cores de maneira mais robusta e precisa. Se as imagens de treinamento forem muito similares em termos de cor ou iluminação, o modelo pode ter dificuldades em generalizar suas previsões para imagens novas ou variadas.

Em resumo, a qualidade das imagens e a variedade de cores presentes nelas têm um impacto direto na capacidade da rede neural de fazer previsões corretas. Quanto mais variado e representativo for o conjunto de imagens em termos de cores, melhor será o desempenho do modelo ao identificar as cores predominantes em novas imagens.

---

### Contribuições

Se você deseja contribuir para este projeto, aqui uma ideia de melhoria:

- **Puxar imagens pela web**:  
   Uma sugestão seria adicionar uma funcionalidade para puxar imagens diretamente da web, utilizando bibliotecas como `requests` para baixar imagens de URLs. Isso poderia permitir que os usuários façam upload de imagens sem precisar salvá-las localmente primeiro. 

