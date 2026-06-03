
# FATORES SOCIOECONÔMICOS DA VIOLÊNCIA LETAL URBANA EM MUNICIPIOS BRASILEIROS




## PASSO A PASSO

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
dnf install python3.11
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