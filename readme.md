
### FATORES SOCIOECONÔMICOS DA VIOLÊNCIA LETAL URBANA: 
### Uma Análise Multinível e de Aprendizado de Máquina em Municipios Brasileiros

### RESUMO
A violência letal urbana no Brasil permanece como um dos principais desafios sociais, refletindo-se em elevadas taxas de homicídios e na necessidade de investigar suas raízes socioeconômicas. Este estudo analisou 282 municípios brasileiros para identificar a contribuição de indicadores socioeconômicos na explicação das taxas de mortes violentas intencionais em 2010. Para mitigar a multicolinearidade, utilizou-se Análise de Componentes Principais [PCA], que condensou 18 variáveis em 5 indicadores complexos: Pobreza e Desigualdade, Educação, Emprego, Infraestrutura e População. Em seguida, aplicaram-se cinco modelos estatísticos e de aprendizado de máquina. Três apresentaram baixo desempenho, mas dois se destacaram: a Regressão Multinível e o Random Forest. O modelo multinível apresentou poder explicativo relevante (R² ≈ 0,617), evidenciando que População e Emprego possuem os maiores coeficientes, seguidos por efeitos marginais de Educação e Infraestrutura. Esses resultados sugerem que a densidade populacional e o desemprego estão positivamente associados ao aumento da violência letal, enquanto a educação exerce efeito redutor e a infraestrutura se mostrou marginalmente relacionada ao aumento, possivelmente por refletir a concentração em grandes centros urbanos. Já o Random Forest obteve melhor ajuste (R² ≈ 0,683), com importâncias mais equilibradas, sem direção dos efeitos, mantendo acurácia em previsões posteriores. O contraste entre os modelos mostra que, enquanto a abordagem multinível destaca relações lineares entre emprego, população e violência, o Random Forest evidencia interações não lineares complexas e duradouras, reforçando a necessidade de políticas que combinem estratégias voltadas ao emprego, urbanização e desigualdade.


### RESULTADOS


![alt text](img_docs/rlmm-bc.png)

![alt text](img_docs/image.png)


|asas| sasasasas| 
---|---|---|---|


## Reproduzindo

### 1. Criando o ambiente

É necessário que o python 3.11 esteja instalado na máquina que for rodar o projeto. Em caso de não estar, será necessário instalar:


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

#### Linux (via dnf ou apt)

```bash
# Fedora / RHEL
sudo dnf install python3.11 python3.11-devel -y

# Debian / Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Arch Linux
yay -S python311
```

Supondo que o python3.11 está instalado no ambiente alvo.

```bash
# cria um ambiente 
python3.11 -m venv fatores_violencia_letal

# ativa o ambiente
source fatores_violencia_letal/bin/activate

# instala as dependências listadas no txt
pip install --no-cache-dir -r requirements.txt
```