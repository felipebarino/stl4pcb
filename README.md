# Conversão de Gerber para STL

## Introdulção e objetivo

Para fabricação de PCBs usando a impressora 3d de resina para expor o filme fotossensível, é necessário obter um arquivo STL contendo os traços de cobre. **STL4PCB** é uma ferramenta Python para converter arquivos Gerber de PCB (camadas de cobre, ilhas, trilhas) em modelos 3D no formato STL. O arquivo de saída deve ser fatiado no software da impressora 3D (25~35 segundos para exposição do filme).

## Como instalar

1. Clone o repositório ou baixe os arquivos.
2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Como usar

Execute o script via linha de comando apontando para o seu arquivo Gerber.

### Uso Básico
Gera um STL com configurações padrão (método multipolygon, espessura 0.035mm).

```bash
python main.py caminho/para/arquivo.gbr
```

### Opções Avançadas
```bash
python main.py input.gbr -o output.stl --method multipolygon --dpmm 100 --thickness 0.035
```

| Argumento | Descrição | Padrão |
| :--- | :--- | :--- |
| `input_file` | Caminho para o arquivo Gerber de entrada (`.gbr`). | (Obrigatório) |
| `-o`, `--output` | Caminho para o arquivo STL de saída. | output.stl |
| `--method` | Algoritmo de geração: `multipolygon` (melhor qualidade) ou `pixel` (mais robusto). | `multipolygon` |
| `--dpmm` | Pontos por milímetro (Resolução). Maior = mais suave e mais lento. | `60` |
| `--thickness` | Espessura da camada de cobre em mm (ex: 1oz = 0.035mm). | `0.035` |

#### Diferença entre os Métodos

- **Multipolygon:** Extrai os contornos da imagem renderizada e usa a biblioteca `Shapely` para triangular a geometria. Gera arquivos STL leves e com paredes lisas, perfeitos para trilhas curvas e ilhas complexas.

- **Pixel:** Trata a imagem como um mapa de bits e cria um bloco sólido para cada pixel branco. Gera arquivos STL muito pesados e serrilhados ("estilo Minecraft"), mas funciona onde a triangulação vetorial pode falhar.


## Como contribuir

### Como reportar problemas

Encontrou um erro ou o STL gerado está incorreto? Ajude-nos a melhorar:

1. **Verifique Issues existentes:** Antes de criar um novo relatório, olhe a aba [Issues](../../issues) para ver se o problema já foi relatado.
2. **Abra uma nova Issue:** Se o problema for novo, clique em "New Issue".
3. **Seja específico:**
   - Descreva o que aconteceu versus o que deveria acontecer.
   - **Anexe logs:** Copie a saída do terminal (preferencialmente com a flag `--verbose`).
   - **Exemplos:** Se possível, anexe o arquivo `.gbr` que causou o erro e um print do STL resultante.
   - Informe o comando exato utilizado (ex: `python main.py file.gbr --method pixel`).

### As branches

Para manter o projeto organizado, utilizamos o seguinte padrão de branches. Por favor, siga-o ao criar sua contribuição:

*   **`main`**: Contém a versão estável e testada do projeto (Produção). Os commits aqui são feitos apenas via Merge Request de versões finalizadas.
*   **`dev`**: Branch principal de desenvolvimento. Todas as novas contribuições devem ser direcionadas (Pull Request) para cá primeiro.
*   **Branches de Trabalho (Temporárias):**
    *   `feat/nome-da-feature`: Para novas funcionalidades (ex: `feat/novo-algoritmo-triangulacao`).
    *   `bug/nome-do-erro`: Para correções de bugs (ex: `bug/fix-guard-ring-hole`).
    *   `doc/o-que-mudou`: Para alterações apenas na documentação (ex: `doc/atualizacao-readme`).

### Como enviar alterações (Pull Request)

1. Faça um **Fork** deste repositório.
2. Clone o seu fork para sua máquina.
3. Crie uma branch seguindo o padrão acima (partindo da `dev`):
   ```bash
   git checkout -b feat/minha-melhoria dev
4. Faça suas alterações e teste gerando alguns STLs.
5. Envie para o seu repositório (Push):
    ```bash
    git push origin feat/minha-melhoria
    ```
6. No GitHub, abra um Pull Request (PR) da sua branch para a branch `dev` do repositório original.
7. Aguarde a revisão do código.

### Como funciona o processamento

1.  **Leitura do Gerber (`pygerber`):** O script lê o arquivo vetorial `.gbr` e o renderiza em uma imagem de alta resolução (rasterização).
2.  **Criação de Máscara (Binarização):** A imagem é convertida em preto e branco. Onde é branco, existe cobre; onde é preto, é o substrato da placa.
3.  **Processamento da Geometria:**
    *   **Método Pixel:** Percorre cada pixel branco da imagem e cria um pequeno bloco 3D (voxel) para ele. É a abordagem "força bruta".
    *   **Método Multipolygon:**
        1.  Detecta as bordas (contornos) das áreas brancas.
        2.  Usa a XOR para identificar o que é borda externa e o que é furo.
        3.  Cria polígonos vetoriais usando a biblioteca `Shapely`.
        4.  Triangula a superfície desses polígonos.
4.  **Geração do STL (`numpy-stl`):** Os triângulos gerados (seja pelos voxels ou pelos polígonos) são listados e salvos no formato binário STL.
5.  **Visualização (`pyvista`):** Opcionalmente, uma janela 3D abre o arquivo resultante para conferência visual.