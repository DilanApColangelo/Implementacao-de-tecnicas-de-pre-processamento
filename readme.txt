# ================== README ==================
# Este programa implementa funcionalidades de processamento de imagem, incluindo:
# 1. Transformação de tons de cinza com quantização baseada em clusterização (agrupamento uniforme).
# 2. Subtração de imagens para detecção de objeto (corpo) e realce com retângulo.
# 3. Implementação e comparação de filtro high-boost com filtro passa-alta.
# 4. Demonstração do ganho computacional do Teorema da Convolução (filtragem espacial vs. frequência).
#
# Dependências:
# - Python 3.x
# - OpenCV (cv2): pip install opencv-python
# - NumPy: pip install numpy
# - Matplotlib: pip install matplotlib
#
# Como executar:
# 1. Salve este código como um arquivo Python (ex: main.py).
# 2. Certifique-se de ter as imagens necessárias nos caminhos especificados ou atualize os caminhos no código.
#    - Para a Funcionalidade 1, 3 e 4: 'input_image.jpg' (uma imagem colorida qualquer).
#    - Para a Funcionalidade 2: 'background.jpg' (imagem do fundo) e 'person_with_background.jpg' (imagem da pessoa no fundo).
#      A câmera deve estar fixa entre as capturas.
# 3. Execute o script a partir do terminal: python main.py
#
# O programa exibirá as imagens resultantes de cada funcionalidade usando Matplotlib
# e imprimirá informações relevantes (como tempos de processamento) no console.
#
# Nota sobre "executável":
# Este script Python (.py) é o "executável". Se um executável standalone (ex: .exe)
# for necessário, ferramentas como PyInstaller podem ser usadas, mas isso está
# fora do escopo deste script.
# ============================================