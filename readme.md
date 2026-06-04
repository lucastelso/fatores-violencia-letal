
### FATORES SOCIOECONÔMICOS DA VIOLÊNCIA LETAL URBANA
### Uma Análise Multinível e de Aprendizado de Máquina em Municipios Brasileiros

#### RESUMO GERAL
A violência letal urbana no Brasil permanece como um dos principais desafios sociais, refletindo-se em elevadas taxas de homicídios e na necessidade de investigar suas raízes socioeconômicas. Este estudo analisou 282 municípios brasileiros para identificar a contribuição de indicadores socioeconômicos na explicação das taxas de mortes violentas intencionais em 2010. Para mitigar a multicolinearidade, utilizou-se Análise de Componentes Principais [PCA], que condensou 18 variáveis em 5 indicadores complexos: Pobreza e Desigualdade, Educação, Emprego, Infraestrutura e População. Em seguida, aplicaram-se cinco modelos estatísticos e de aprendizado de máquina. Três apresentaram baixo desempenho, mas dois se destacaram: a Regressão Multinível e o Random Forest. O modelo multinível apresentou poder explicativo relevante (R² ≈ 0,617), evidenciando que População e Emprego possuem os maiores coeficientes, seguidos por efeitos marginais de Educação e Infraestrutura. Esses resultados sugerem que a densidade populacional e o desemprego estão positivamente associados ao aumento da violência letal, enquanto a educação exerce efeito redutor e a infraestrutura se mostrou marginalmente relacionada ao aumento, possivelmente por refletir a concentração em grandes centros urbanos. Já o Random Forest obteve melhor ajuste (R² ≈ 0,683), com importâncias mais equilibradas, sem direção dos efeitos, mantendo acurácia em previsões posteriores. O contraste entre os modelos mostra que, enquanto a abordagem multinível destaca relações lineares entre emprego, população e violência, o Random Forest evidencia interações não lineares complexas e duradouras, reforçando a necessidade de políticas que combinem estratégias voltadas ao emprego, urbanização e desigualdade.


#### PRINCIPAIS RESULTADOS

Os resultados deste trabalho validam a hipótese central da pesquisa de que a violência pode ser modelada como uma função dos indicadores socioeconômicos, conforme demonstrado pelos modelos testados. Entre eles, destacaram-se os modelos RLMM-BC e RFR, ambos com capacidade de explicar mais de 60% da variação de violência no ano de 2010 e até o ano de 2016, no caso do RFR. Esta capacidade preditiva reforça a importância de tais indicadores não apenas para o estudo acadêmico da violência, mas também para a formulação de políticas públicas baseadas em evidências

Os resultados do modelo RLMM-BC sugerem que o volume populacional urbano e o desemprego estão diretamente associados ao aumento da incidência de violência letal nos municípios e devem ser levados em conta na formulação de políticas públicas. No caso do RFR, a violência é explicada pelo conjunto de preditores de forma combinada, conseguindo capturar relações complexas que se mantiveram relevantes ao longo do tempo.

#### Comparação dos coeficientes normalizados dos modelos paramétricos
| FATOR | RLM | p >\|t\| | Log-RLM | p >\|t\| | RLM-BC | p >\|t\| | RLMM-BC | p >\|t\| |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Intercepto | 31,608 | 0,00 | 24,483 | 0,00 | 25,978 | 0,00 | 33,473 | 0,00 |
| Pobreza e Desigualdade | 1,170 | 0,36 | 0,058 | 0,18 | 0,117 | 0,17 | -0,094 | 0,38 |
| Infraestrutura | 0,664 | 0,56 | 0,047 | 0,22 | 0,087 | 0,25 | 0,185 | 0,01 |
| Educação | -2,046 | 0,04 | -0,064 | 0,04 | -0,125 | 0,04 | -0,148 | 0,01 |
| Emprego | -10,84 | 0,00 | -0,297 | 0,00 | -0,534 | 0,00 | -0,645 | 0,00 |
| População | 6,042 | 0,00 | 0,235 | 0,00 | 0,487 | 0,00 | 0,732 | 0,00 |
| Group Var. | N/A | N/A | N/A | N/A | N/A | N/A | 0,127 | 0,01 |
| Group x Emprego cov. | N/A | N/A | N/A | N/A | N/A | N/A | 0,078 | 0,88 |
| Emprego (nível 2) Var. | N/A | N/A | N/A | N/A | N/A | N/A | 0,073 | 0,58 |
---
![RLMM-BC](img_docs/rlmm-bc.png)

![RANDOM FOREST](img_docs/image.png)

É importante reforçar que os modelos foram treinados com dados de 2010 e a queda observada de sua performance em anos mais recentes indica que as dinâmicas da violência e as condições socioeconômicas se alteraram ao longo desses anos. Portanto, a futura disponibilização dos dados do Censo de 2022 permitirá a realização de novas pesquisas que possam reavaliar os pesos dos fatores socioeconômicos para a década de 2020.


---

## Reproduzindo

### 1. Criando o ambiente

É necessário que o python 3.11 esteja instalado na máquina que for rodar o projeto. Em caso de não estar, será necessário instalar:

### 1.1. Instalando o Python 3.11

#### Windows (via PowerShell)

```powershell
wget
https://python.org
-OutFile python-3.11.9-adm64.exe # Ver a melhor versão
.\python-3.11.9-amd64.exe /quiet InstallAllUsers=1 PrependPath=1


# verificar instalação com
python --version # deve retornar 3.11.9

```

OU

```powershell
# ALTERNATIVAMENTE
winget install Python.Python.3.11
```

#### Linux (via _package manager_)

```bash
# Fedora / RHEL
sudo dnf install python3.11 python3.11-devel -y

# Debian / Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Arch Linux
yay -S python311
```

### 1.2 Reproduzindo o ambiente

Supondo que o python3.11 está instalado no ambiente alvo, pode-se criar um ambiente chamado "fatores_violencia_letal", ativá-lo e instalar as dependências listas em `requirements.txt`. É importante que isto seja feito na raíz do projeto.

```bash
# cria um ambiente 
python3.11 -m venv fatores_violencia_letal

# ativa o ambiente
source fatores_violencia_letal/bin/activate

# instala as dependências listadas no txt
pip install --no-cache-dir -r requirements.txt
```

