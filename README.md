# ANATO3D

## Descrição

Anato3D é um protótipo acadêmico de aplicação web para visualização interativa da anatomia humana em 3D. A ideia central é usar a câmera do navegador para detectar uma pessoa e estimar a posição aproximada de regiões anatômicas, como cabeça, tórax e tronco, a partir de landmarks corporais.

## Objetivo

O objetivo inicial do projeto é validar a etapa de detecção corporal e visualização dos pontos sobre a imagem da câmera, sem ainda implementar realidade aumentada ou modelos 3D em cena.

> O sistema realiza uma estimativa da localização anatômica dos órgãos a partir de pontos corporais identificados pela câmera. Os órgãos representados são modelos tridimensionais posicionados virtualmente e não correspondem à visualização direta dos órgãos internos do indivíduo.

## Tecnologias utilizadas

- HTML5
- CSS3
- JavaScript moderno (ES6+)
- MediaPipe Pose Landmarker para Web
- API de câmera do navegador (`navigator.mediaDevices.getUserMedia()`)

## Requisitos

- Navegador moderno com suporte a câmera
- Servidor local para permitir acesso à câmera em ambiente seguro
- Conexão com a internet para carregar o modelo de detecção do MediaPipe

## Instalação e execução

1. Abra a pasta do projeto no VS Code.
2. Inicie um servidor local na pasta do projeto.
3. Acesse a URL:
   - `http://localhost:8000`
   - ou `http://127.0.0.1:8000`

### Exemplo usando Python

```bash
cd c:\Users\251016600009\Documents\Anato3D
python -m http.server 8000
```

Depois abra o navegador e acesse:

```text
http://localhost:8000
```

## Estrutura do projeto

```text
anato3d/
├── index.html
├── style.css
├── app.js
├── README.md
├── assets/
│   ├── modelos/
│   │   └── .gitkeep
│   └── originais/
│       └── .gitkeep
└──
```

## Funcionalidades da etapa atual

- acesso à câmera do computador;
- detecção corporal com MediaPipe;
- desenho dos pontos e conexões do corpo sobre o vídeo;
- indicação visual de que o corpo foi detectado;
- interface responsiva para execução em navegador.

## Como testar a webcam

1. Clique no botão "INICIAR CÂMERA".
2. Permita o acesso à câmera quando o navegador solicitar.
3. Posicione a pessoa em frente à câmera.
4. Verifique se os pontos do corpo aparecem sobre o vídeo.

## Limitações conhecidas

- Esta fase não implementa modelos 3D ainda.
- A detecção é estimativa anatômica baseada em landmarks corporais.
- Não há visualização de órgãos internos reais.
- O comportamento depende da qualidade da câmera, da iluminação e do posicionamento do corpo.

## Próximos passos

- estimar posições de cabeça, tórax e tronco;
- carregar modelos 3D (GLB);
- alinhar os órgãos virtualmente sobre o corpo;
- adicionar controles de visibilidade e escala;
- preparar a experiência para uso em Android e iPhone via WebXR/AR.
