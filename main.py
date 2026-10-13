import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Funções Auxiliares ---
def carregar_imagem(caminho, colorida=True):
    """Carrega uma imagem do caminho especificado."""
    if colorida:
        img = cv2.imread(caminho, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar a imagem: {caminho}")
        print("Certifique-se de que o arquivo de imagem existe no caminho especificado.")
        print("Crie arquivos de imagem dummy se necessário para testar:")
        print("  - input_image.jpg (colorida, ex: 640x480)")
        print("  - background.jpg (colorida, ex: 640x480)")
        print("  - person_with_background.jpg (colorida, ex: 640x480)")
        print("Pressione Enter para tentar criar imagens dummy...")
        input() # Pausa para o usuário ler
        try:
            # Criar imagens dummy se não existirem para permitir a execução
            dummy_color = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            dummy_gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

            if caminho == 'input_image.jpg' and not cv2.imwrite(caminho, dummy_color):
                print(f"Falha ao criar {caminho}")
            elif caminho == 'background.jpg' and not cv2.imwrite(caminho, dummy_color):
                 print(f"Falha ao criar {caminho}")
            elif caminho == 'person_with_background.jpg' and not cv2.imwrite(caminho, dummy_color):
                 print(f"Falha ao criar {caminho}")

            print(f"Imagem dummy '{caminho}' criada. Por favor, substitua por uma imagem real para melhores resultados.")
            img = cv2.imread(caminho, cv2.IMREAD_COLOR if colorida else cv2.IMREAD_GRAYSCALE)
            if img is None: # Checagem final
                raise FileNotFoundError(f"Ainda não foi possível carregar ou criar {caminho}")

        except Exception as e:
            print(f"Erro crítico ao tentar criar/carregar imagem dummy: {e}")
            return None # Retorna None se falhar em criar ou carregar
    return img

def exibir_imagens(titulos, imagens, cmap_list=None):
    """Exibe uma lista de imagens com seus títulos usando Matplotlib."""
    num_imagens = len(imagens)
    if cmap_list is None:
        cmap_list = ['gray'] * num_imagens
        for i, img in enumerate(imagens):
            if len(img.shape) == 3: # Se for colorida
                 cmap_list[i] = None # imshow lida com BGR -> RGB
                 imagens[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    plt.figure(figsize=(15, 5 * ((num_imagens +1) // 2)))
    for i in range(num_imagens):
        plt.subplot(1, num_imagens, i + 1)
        plt.title(titulos[i])
        plt.imshow(imagens[i], cmap=cmap_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Funcionalidade 1: Transformação de Tons de Cinza e Clusterização ---
def funcionalidade_1_tons_de_cinza_clusterizado(caminho_imagem):
    """
    Converte uma imagem colorida para tons de cinza e agrupa os tons de cinza
    a cada 4 grupos (quantização).
    """
    print("\n--- Iniciando Funcionalidade 1: Tons de Cinza e Quantização ---")
    img_colorida = carregar_imagem(caminho_imagem)
    if img_colorida is None:
        return

    # Converter para tons de cinza (OpenCV usa o método de luminosidade ponderada)
    img_cinza = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)

    # Agrupar tons de cinza (Quantização)
    # original_levels = 256 (0-255)
    # new_levels = 256 / 4 = 64
    # new_value = floor(old_value / 4)
    # Para visualização, podemos escalar de volta para a faixa 0-252 (com 64 níveis distintos)
    # ou usar os valores 0-63 diretamente.
    # Optamos por escalar para melhor visualização na faixa 0-255.

    # img_quantizada_0_63 = img_cinza // 4 # Resulta em valores de 0 a 63
    img_quantizada_0_252 = (img_cinza // 4) * 4 # Resulta em valores 0, 4, 8, ..., 252

    print("Funcionalidade 1 concluída.")
    exibir_imagens(
        ["Original Colorida", "Tons de Cinza", "Tons de Cinza Quantizados (64 níveis)"],
        [img_colorida, img_cinza, img_quantizada_0_252]
    )
    return img_cinza # Retornar para uso posterior se necessário

# --- Funcionalidade 2: Subtração de Imagens e Realce ---
def funcionalidade_2_subtracao_realce(caminho_fundo, caminho_pessoa, limiar_empirico=30):
    """
    Aplica subtração entre imagem de fundo e imagem com pessoa, binariza,
    e desenha um retângulo na área do corpo detectado.
    """
    print("\n--- Iniciando Funcionalidade 2: Subtração de Imagens ---")
    img_fundo_color = carregar_imagem(caminho_fundo)
    img_pessoa_color = carregar_imagem(caminho_pessoa)

    if img_fundo_color is None or img_pessoa_color is None:
        return

    # Converter ambas para tons de cinza
    img_fundo_cinza = cv2.cvtColor(img_fundo_color, cv2.COLOR_BGR2GRAY)
    img_pessoa_cinza = cv2.cvtColor(img_pessoa_color, cv2.COLOR_BGR2GRAY)

    # Aplicar subtração (diferença absoluta)
    img_diferenca = cv2.absdiff(img_pessoa_cinza, img_fundo_cinza)

    # Binarizar a imagem de diferença
    _, img_binarizada = cv2.threshold(img_diferenca, limiar_empirico, 255, cv2.THRESH_BINARY)

    # Encontrar contornos na imagem binarizada
    contornos, _ = cv2.findContours(img_binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_pessoa_com_retangulo = img_pessoa_color.copy()
    if contornos:
        # Assumir que o maior contorno é a pessoa
        maior_contorno = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior_contorno) > 500: # Adicionar um filtro de área mínima
            x, y, w, h = cv2.boundingRect(maior_contorno)
            cv2.rectangle(img_pessoa_com_retangulo, (x, y), (x + w, y + h), (0, 0, 255), 2) # Vermelho
            print(f"Corpo detectado em: x={x}, y={y}, w={w}, h={h}")
        else:
            print("Nenhum contorno significativo encontrado após binarização.")
    else:
        print("Nenhum contorno encontrado após binarização.")

    print("Funcionalidade 2 concluída.")
    exibir_imagens(
        ["Fundo", "Pessoa Original", "Diferença", "Binarizada", "Pessoa Destacada"],
        [img_fundo_color, img_pessoa_color, img_diferenca, img_binarizada, img_pessoa_com_retangulo]
    )

# --- Funcionalidade 3: Filtro High-Boost ---
def _calcular_detalhes_high_pass(img_cinza, kernel_size=(5,5)):
    """Calcula os detalhes high-pass (Original - LowPass) em float."""
    img_float = img_cinza.astype(np.float32)
    blur_float = cv2.GaussianBlur(img_float, kernel_size, 0)
    high_pass_details_float = img_float - blur_float
    return high_pass_details_float

def filtro_passa_alta(img_cinza, kernel_size=(5,5)):
    """Aplica um filtro passa-alta simples (Original - LowPass) e normaliza para exibição."""
    high_pass_details_float = _calcular_detalhes_high_pass(img_cinza, kernel_size)
    # Normalizar para exibição (0-255)
    img_passa_alta_display = cv2.normalize(high_pass_details_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_passa_alta_display

def filtro_high_boost_custom(img_cinza, A, kernel_size=(5,5)):
    """
    Aplica um filtro high-boost customizado.
    HighBoost = A * Original - LowPass(Original)
    ou equivalentemente: HighBoost = Original + (A-1) * HighPassDetails(Original)
    """
    img_float = img_cinza.astype(np.float32)
    blur_float = cv2.GaussianBlur(img_float, kernel_size, 0)

    # Fórmula: HighBoost = A * Original - LowPass
    high_boost_float = A * img_float - blur_float

    # Clip e converte para uint8
    high_boost_uint8 = np.clip(high_boost_float, 0, 255).astype(np.uint8)
    return high_boost_uint8

def funcionalidade_3_high_boost(caminho_imagem, fator_A=1.5):
    """
    Cria e aplica um filtro high-boost, comparando com filtro passa-alta.
    """
    print("\n--- Iniciando Funcionalidade 3: Filtro High-Boost ---")
    img_colorida = carregar_imagem(caminho_imagem)
    if img_colorida is None:
        return

    img_cinza = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro passa-alta
    img_passa_alta = filtro_passa_alta(img_cinza)

    # Aplicar filtro high-boost customizado
    img_high_boost = filtro_high_boost_custom(img_cinza, fator_A)

    print(f"Fator A para High-Boost: {fator_A}")
    print("Análise:")
    print("- O filtro Passa-Alta realça as bordas e detalhes de alta frequência, mas pode perder contraste geral e escurecer a imagem.")
    print("- O filtro High-Boost também realça as bordas, mas, devido ao fator A > 1 multiplicando a imagem original")
    print("  (ou adicionando uma fração da original de volta ao resultado do passa-alta),")
    print("  ele preserva melhor o brilho e contraste originais, resultando em uma imagem mais nítida e menos 'vazia'.")
    print("Funcionalidade 3 concluída.")

    exibir_imagens(
        ["Original Cinza", f"Passa-Alta (Kernel {(5,5)})", f"High-Boost (A={fator_A}, Kernel {(5,5)})"],
        [img_cinza, img_passa_alta, img_high_boost]
    )

# --- Funcionalidade 4: Teorema da Convolução ---
def funcionalidade_4_teorema_convolucao(caminho_imagem, kernel_size=31):
    """
    Demonstra o ganho computacional do Teorema da Convolução.
    Compara o tempo de filtragem no domínio espacial vs. domínio da frequência.
    """
    print("\n--- Iniciando Funcionalidade 4: Teorema da Convolução ---")
    img_colorida = carregar_imagem(caminho_imagem)
    if img_colorida is None:
        return

    img_cinza = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)
    img_cinza_float = img_cinza.astype(np.float32) / 255.0 # Normalizar para float32

    # Criar um kernel (ex: kernel de média ou Gaussiano)
    # Usaremos um kernel Gaussiano, que é separável e simétrico
    # kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    sigma = kernel_size / 6.0 # Heurística para sigma
    kernel_gauss_1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel_gauss_1d, kernel_gauss_1d)
    kernel = kernel.astype(np.float32)


    print(f"Tamanho do Kernel: {kernel_size}x{kernel_size}")

    # 1. Convolução no Domínio Espacial
    start_time_espacial = time.perf_counter()
    filtrada_espacial = cv2.filter2D(img_cinza_float, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    end_time_espacial = time.perf_counter()
    tempo_espacial = end_time_espacial - start_time_espacial
    print(f"Tempo de convolução espacial: {tempo_espacial:.6f} segundos")

    # 2. Convolução no Domínio da Frequência
    start_time_frequencia = time.perf_counter()

    h, w = img_cinza_float.shape
    # Otimizar tamanho para DFT
    opt_H = cv2.getOptimalDFTSize(h + kernel_size - 1) # Tamanho para convolução linear completa
    opt_W = cv2.getOptimalDFTSize(w + kernel_size - 1)

    # Pad da imagem
    padded_img = cv2.copyMakeBorder(img_cinza_float, 0, opt_H - h, 0, opt_W - w, cv2.BORDER_CONSTANT, value=0)

    # Pad do kernel
    padded_kernel = np.zeros((opt_H, opt_W), dtype=np.float32)
    # Posicionar o kernel no canto superior esquerdo. O centro do kernel é (kh//2, kw//2)
    # Para convolução, o kernel precisa ser "virado" ou posicionado corretamente.
    # cv2.filter2D faz correlação. Para convolução via FFT, o kernel também é transformado.
    # Simplesmente colocar no canto superior esquerdo e depois ajustar o resultado é comum.
    kh, kw = kernel.shape
    padded_kernel[0:kh, 0:kw] = kernel
    # Se o kernel não for simétrico e o centro não for (0,0), ifftshift(kernel) antes do DFT é necessário.
    # Para kernel Gaussiano, que é simétrico, e com processamento cuidadoso, o posicionamento no canto superior esquerdo funciona.

    # DFT da imagem e do kernel
    dft_img = cv2.dft(padded_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_kernel = cv2.dft(padded_kernel, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Multiplicação no domínio da frequência
    dft_filtrada = cv2.mulSpectrums(dft_img, dft_kernel, flags=0, conjB=False) # False para convolução

    # Inversa DFT
    # DFT_SCALE normaliza o resultado. DFT_REAL_OUTPUT retorna uma imagem real de canal único.
    img_filtrada_freq_complex = cv2.idft(dft_filtrada, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Cortar a imagem para o tamanho original
    # O resultado da convolução via FFT é deslocado se o kernel não for devidamente centralizado e "shiftado" antes da FFT.
    # Com o kernel no canto superior esquerdo, o resultado válido começa em (kh//2, kw//2) aprox.
    # No entanto, se o padding foi para 'full convolution', o resultado útil pode ser maior.
    # Para um resultado 'same' como filter2D, precisamos do crop correto.
    # Com padding para (N+M-1), o resultado é a convolução linear.
    # Para replicar filter2D (que faz 'same' por padrão com BORDER_REPLICATE), o crop é:
    center_x, center_y = kh // 2, kw // 2
    filtrada_frequencia = img_filtrada_freq_complex[center_y : center_y + h, center_x : center_x + w]
    # filtrada_frequencia = img_filtrada_freq_complex[0:h, 0:w] # Se o kernel fosse shiftado para o centro antes do DFT

    end_time_frequencia = time.perf_counter()
    tempo_frequencia = end_time_frequencia - start_time_frequencia
    print(f"Tempo de convolução no domínio da frequência: {tempo_frequencia:.6f} segundos")

    print("\nAnálise dos Tempos:")
    print("- A convolução espacial é geralmente mais rápida para kernels pequenos.")
    print("- A convolução no domínio da frequência (usando FFT) tende a ser mais rápida para kernels grandes.")
    print("  Isso ocorre porque a complexidade da FFT é O(N log N), enquanto a convolução espacial direta é O(N*M),")
    print("  onde N é o número de pixels da imagem e M é o número de pixels do kernel.")
    print("  A multiplicação no domínio da frequência é O(N).")
    print(f"  Neste caso (kernel {kernel_size}x{kernel_size}):")
    if tempo_frequencia < tempo_espacial:
        print("  O domínio da frequência foi mais rápido.")
    else:
        print("  O domínio espacial foi mais rápido ou comparável.")
    print("  Experimente variar o `kernel_size` para observar a mudança na performance relativa.")
    print("Funcionalidade 4 concluída.")

    # Normalizar para exibição se necessário (já estão em float 0-1 ou similar)
    filtrada_espacial_disp = np.clip(filtrada_espacial * 255, 0, 255).astype(np.uint8)
    filtrada_frequencia_disp = np.clip(filtrada_frequencia * 255, 0, 255).astype(np.uint8)


    exibir_imagens(
        ["Original Cinza", f"Filtrada Espacial (Kernel {kernel_size}x{kernel_size})", f"Filtrada Frequência (Kernel {kernel_size}x{kernel_size})"],
        [img_cinza, filtrada_espacial_disp, filtrada_frequencia_disp]
    )


# --- Execução Principal ---
if __name__ == "__main__":
    # Caminhos das imagens (substitua pelos seus caminhos ou crie imagens dummy)
    caminho_imagem_principal = 'input_image.jpg'
    caminho_imagem_fundo = 'background.jpg'
    caminho_imagem_pessoa = 'person_with_background.jpg'

    # Criar imagens dummy se não existirem para evitar erro inicial
    for p in [caminho_imagem_principal, caminho_imagem_fundo, caminho_imagem_pessoa]:
        if cv2.imread(p) is None: # Se a imagem não puder ser lida
            print(f"Tentando criar imagem dummy para '{p}' pois não foi encontrada.")
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(p, dummy_img)


    # Executar Funcionalidade 1
    img_cinza_f1 = funcionalidade_1_tons_de_cinza_clusterizado(caminho_imagem_principal)

    # Executar Funcionalidade 2
    # Ajuste o limiar_empirico conforme necessário para suas imagens
    funcionalidade_2_subtracao_realce(caminho_imagem_fundo, caminho_imagem_pessoa, limiar_empirico=25)

    # Executar Funcionalidade 3
    # Ajuste o fator_A conforme necessário
    funcionalidade_3_high_boost(caminho_imagem_principal, fator_A=2.0)

    # Executar Funcionalidade 4
    # Experimente diferentes tamanhos de kernel, ex: 5, 15, 31, 51
    funcionalidade_4_teorema_convolucao(caminho_imagem_principal, kernel_size=31)
    # funcionalidade_4_teorema_convolucao(caminho_imagem_principal, kernel_size=5) # Para comparar com kernel pequeno

    print("\nTodas as funcionalidades foram executadas.")
    print("Verifique as janelas do Matplotlib para os resultados visuais e o console para as análises.")