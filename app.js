import { FilesetResolver, PoseLandmarker, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest';
import { MTLLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/loaders/OBJLoader.js';

const video = document.getElementById('video');
const poseCanvas = document.getElementById('poseCanvas');
const statusPill = document.getElementById('statusPill');
const startCameraBtn = document.getElementById('startCameraBtn');
const sceneContainer = document.getElementById('sceneContainer');

const ctx = poseCanvas.getContext('2d');

const FACE_OUTLINE = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
  377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
  67, 109, 104, 71, 63, 105, 66, 107, 55, 65, 52, 53, 46
];

const FACE_GUIDE_POINTS = [1, 33, 61, 291, 199, 152, 263, 468, 454, 234];

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [8, 10], [10, 12], [11, 12],
  [11, 13], [13, 15], [15, 17],
  [15, 19], [15, 21], [17, 19],
  [11, 23], [23, 25], [25, 27],
  [27, 29], [29, 31], [23, 24],
  [24, 26], [26, 28], [28, 30],
  [19, 20], [20, 18], [18, 16],
  [16, 14], [14, 12], [12, 13],
  [13, 14], [14, 16], [17, 18]
];

let poseLandmarker = null;
let faceLandmarker = null;
let animationFrameId = null;
let isCameraActive = false;
let threeScene = null;
let camera = null;
let renderer = null;
let headGroup = null;
let brainMesh = null;
let headPose = { yaw: 0, pitch: 0, roll: 0 };
const BRAIN_MODEL_BASE_URL = './assets/brain/';

function smoothHeadPose(target) {
  headPose.yaw = THREE.MathUtils.lerp(headPose.yaw, target.yaw, 0.12);
  headPose.pitch = THREE.MathUtils.lerp(headPose.pitch, target.pitch, 0.12);
  headPose.roll = THREE.MathUtils.lerp(headPose.roll, target.roll, 0.12);
}

function setupThreeScene() {
  if (threeScene) {
    return;
  }

  threeScene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, sceneContainer.clientWidth / sceneContainer.clientHeight, 0.1, 1000);
  camera.position.set(0, 0, 5.8);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
  renderer.setClearColor(0x000000, 0);
  sceneContainer.appendChild(renderer.domElement);

  const ambientLight = new THREE.AmbientLight(0xffffff, 1.6);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
  directionalLight.position.set(2, 3, 5);

  threeScene.add(ambientLight);
  threeScene.add(directionalLight);

  headGroup = new THREE.Group();

  const skinMaterial = new THREE.MeshStandardMaterial({
    color: 0xf4d2b9,
    roughness: 0.9,
    metalness: 0.05
  });

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(1.18, 48, 48),
    skinMaterial
  );
  head.scale.set(1.08, 1.22, 1.02);

  const jaw = new THREE.Mesh(
    new THREE.SphereGeometry(0.72, 32, 32),
    skinMaterial
  );
  jaw.scale.set(1.05, 0.75, 0.92);
  jaw.position.set(0, -0.86, 0.12);

  const skullCap = new THREE.Mesh(
    new THREE.SphereGeometry(0.82, 28, 28),
    new THREE.MeshStandardMaterial({
      color: 0xf5d7b8,
      transparent: true,
      opacity: 0.7,
      roughness: 0.8
    })
  );
  skullCap.scale.set(1.05, 1.1, 0.9);
  skullCap.position.z = 0.18;

  const brainMaterial = new THREE.MeshStandardMaterial({
    color: 0x7dd3fc,
    emissive: 0x1d4ed8,
    transparent: true,
    opacity: 0.78,
    roughness: 0.45,
    metalness: 0.2
  });

  brainMesh = new THREE.Group();
  const fallbackBrain = new THREE.Mesh(
    new THREE.SphereGeometry(0.8, 28, 28),
    brainMaterial
  );
  fallbackBrain.scale.set(0.9, 0.78, 0.72);
  brainMesh.add(fallbackBrain);
  brainMesh.position.set(0, 0, 0.42);

  const neck = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.3, 0.5, 22),
    skinMaterial
  );
  neck.position.set(0, -1.38, 0);

  headGroup.add(head);
  headGroup.add(jaw);
  headGroup.add(skullCap);
  headGroup.add(brainMesh);
  headGroup.add(neck);
  head.visible = false;
  jaw.visible = false;
  skullCap.visible = false;
  neck.visible = false;
  headGroup.visible = false;

  threeScene.add(headGroup);
  loadBrainModel();

  window.addEventListener('resize', () => {
    if (!sceneContainer) {
      return;
    }

    camera.aspect = sceneContainer.clientWidth / sceneContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
  });
}

async function loadBrainModel() {
  if (!brainMesh) {
    return;
  }

  const materialsLoader = new MTLLoader();
  const objectLoader = new OBJLoader();

  try {
    materialsLoader.setPath(BRAIN_MODEL_BASE_URL);
    const materials = await materialsLoader.loadAsync('Meshy_AI_Human_Brain_Anatomy_0921190125_texture.mtl');
    materials.preload();

    objectLoader.setMaterials(materials);
    objectLoader.setPath(BRAIN_MODEL_BASE_URL);
    const model = await objectLoader.loadAsync('Meshy_AI_Human_Brain_Anatomy_0921190125_texture.obj');
    const bounds = new THREE.Box3().setFromObject(model);
    const size = bounds.getSize(new THREE.Vector3());
    const center = bounds.getCenter(new THREE.Vector3());
    const largestDimension = Math.max(size.x, size.y, size.z) || 1;
    const modelScale = 0.95 / largestDimension;

    model.position.set(-center.x * modelScale, -center.y * modelScale, -center.z * modelScale);
    model.scale.setScalar(modelScale);

    model.traverse((object) => {
      if (!object.isMesh) {
        return;
      }

      object.frustumCulled = false;

      if (object.material) {
        object.material.side = THREE.DoubleSide;
      }
    });

    brainMesh.clear();
    brainMesh.add(model);
    console.log('[Anato3D] Modelo OBJ do cérebro carregado com MTL e textura.');
  } catch (error) {
    console.warn('[Anato3D] Não foi possível carregar o cérebro OBJ. Usando fallback.', error);
  }
}

function updateHeadFromFace(faceLandmarks) {
  if (!headGroup || !brainMesh) {
    return;
  }

  const leftEye = faceLandmarks[33] || faceLandmarks[133];
  const rightEye = faceLandmarks[263] || faceLandmarks[362];
  const nose = faceLandmarks[1] || faceLandmarks[168];
  const chin = faceLandmarks[152] || faceLandmarks[14];
  const forehead = faceLandmarks[10] || faceLandmarks[151];

  if (!leftEye || !rightEye || !nose || !chin || !forehead) {
    headGroup.visible = false;
    return;
  }

  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const eyeCenterY = (leftEye.y + rightEye.y) / 2;
  const eyeDistance = Math.max(0.06, Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y));
  const faceHeight = Math.max(0.12, chin.y - forehead.y);

  // Frontal faces have the nose centered on the eye axis; this avoids false yaw.
  const yaw = THREE.MathUtils.clamp((nose.x - eyeCenterX) / eyeDistance * 1.8, -1, 1);
  const expectedNoseY = eyeCenterY + (faceHeight * 0.34);
  const pitch = THREE.MathUtils.clamp((nose.y - expectedNoseY) / faceHeight * 2.4, -0.8, 0.8);
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  const brainCenterY = forehead.y + ((chin.y - forehead.y) * 0.3);
  const mirroredCenterX = 1 - eyeCenterX;
  const x = (mirroredCenterX - 0.5) * 5.5;
  const y = (0.5 - brainCenterY) * 4.8;
  const skullWidth = eyeDistance * 2.55;
  const scale = THREE.MathUtils.clamp(skullWidth * 2.7, 0.72, 1.35);

  // The camera preview is mirrored, so horizontal head rotation must be mirrored too.
  const targetYaw = THREE.MathUtils.clamp(-yaw * 1.15, -1.15, 1.15);
  const targetPitch = THREE.MathUtils.clamp(pitch, -0.8, 0.8);
  const targetRoll = THREE.MathUtils.clamp(roll * 1.4, -1.1, 1.1);

  smoothHeadPose({ yaw: targetYaw, pitch: targetPitch, roll: targetRoll });

  headGroup.visible = true;
  headGroup.position.set(x, y, 0.8);
  headGroup.rotation.set(headPose.pitch, headPose.yaw, headPose.roll);
  headGroup.scale.setScalar(scale);

  const normalizedYaw = Math.abs(headPose.yaw);
  const orientationLabel = normalizedYaw < 0.3
    ? 'frontal'
    : normalizedYaw < 1.0
      ? 'lateral'
      : 'girando';

  statusPill.textContent = `Cabeça detectada: ${orientationLabel}`;
  statusPill.classList.remove('not-detected');
  statusPill.classList.add('detected');

  if (renderer && threeScene && camera) {
    renderer.render(threeScene, camera);
  }
}

async function initPoseLandmarker() {
  if (poseLandmarker) {
    return poseLandmarker;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    return poseLandmarker;
  } catch (error) {
    console.error('Erro ao inicializar o MediaPipe:', error);
    throw new Error('Não foi possível carregar o detector corporal.');
  }
}

async function initFaceLandmarker() {
  if (faceLandmarker) {
    return faceLandmarker;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    return faceLandmarker;
  } catch (error) {
    console.error('Erro ao inicializar o Face Landmarker:', error);
    throw new Error('Não foi possível carregar o detector facial.');
  }
}

function resizeCanvas() {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;

  poseCanvas.width = width;
  poseCanvas.height = height;
}

function drawSkeleton(landmarks) {
  ctx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);

  ctx.lineWidth = 3;
  ctx.strokeStyle = '#7dd3fc';

  for (const [startIndex, endIndex] of POSE_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];

    if (!start || !end) {
      continue;
    }

    const startX = (1 - start.x) * poseCanvas.width;
    const startY = start.y * poseCanvas.height;
    const endX = (1 - end.x) * poseCanvas.width;
    const endY = end.y * poseCanvas.height;

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  }

  for (const point of landmarks) {
    if (!point) {
      continue;
    }

    const x = (1 - point.x) * poseCanvas.width;
    const y = point.y * poseCanvas.height;

    ctx.beginPath();
    ctx.fillStyle = '#f8fafc';
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawFaceLandmarks(faceLandmarks) {
  if (!faceLandmarks || faceLandmarks.length === 0) {
    return;
  }

  ctx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
  ctx.lineWidth = 2.2;
  ctx.strokeStyle = '#34d399';
  ctx.fillStyle = '#dcfce7';
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  const outline = FACE_OUTLINE.map(index => faceLandmarks[index]).filter(Boolean);
  if (outline.length > 1) {
    ctx.beginPath();
    outline.forEach((point, index) => {
      const x = (1 - point.x) * poseCanvas.width;
      const y = point.y * poseCanvas.height;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.closePath();
    ctx.stroke();
  }

  for (const index of FACE_GUIDE_POINTS) {
    const point = faceLandmarks[index];
    if (!point) {
      continue;
    }

    const x = (1 - point.x) * poseCanvas.width;
    const y = point.y * poseCanvas.height;

    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateStatus(isDetected) {
  if (isDetected) {
    statusPill.textContent = 'Corpo detectado';
    statusPill.classList.remove('not-detected');
    statusPill.classList.add('detected');
    return;
  }

  statusPill.textContent = 'Corpo não detectado';
  statusPill.classList.remove('detected');
  statusPill.classList.add('not-detected');
}

function detectPose() {
  if (!poseLandmarker || !video || video.readyState < 2) {
    return;
  }

  if (faceLandmarker) {
    const faceResult = faceLandmarker.detectForVideo(video, performance.now());
    if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
      return;
    }
  }

  const timestamp = performance.now();
  const result = poseLandmarker.detectForVideo(video, timestamp);

  if (!result.landmarks || result.landmarks.length === 0) {
    updateStatus(false);
    ctx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
    return;
  }

  const landmarks = result.landmarks[0];
  drawSkeleton(landmarks);
  updateStatus(true);
}

function detectFace() {
  if (!faceLandmarker || !video || video.readyState < 2) {
    if (headGroup) {
      headGroup.visible = false;
    }
    return;
  }

  const timestamp = performance.now();
  const result = faceLandmarker.detectForVideo(video, timestamp);

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    if (headGroup) {
      headGroup.visible = false;
      statusPill.textContent = 'Corpo detectado, mas rosto não foi localizado';
      statusPill.classList.remove('detected');
      statusPill.classList.add('not-detected');
    }
    return;
  }

  const face = result.faceLandmarks[0];
  drawFaceLandmarks(face);
  updateHeadFromFace(face);
}

function startDetectionLoop() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  const loop = () => {
    detectPose();
    detectFace();
    animationFrameId = requestAnimationFrame(loop);
  };

  animationFrameId = requestAnimationFrame(loop);
}

async function startCamera() {
  console.log('[Anato3D] Iniciando câmera...');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.error('[Anato3D] getUserMedia indisponível no navegador.');
    updateStatus(false);
    statusPill.textContent = 'Seu navegador não suporta acesso à câmera.';
    startCameraBtn.disabled = false;
    startCameraBtn.textContent = 'INICIAR CÂMERA';
    return;
  }

  startCameraBtn.disabled = true;
  startCameraBtn.textContent = 'SOLICITANDO PERMISSÃO...';
  statusPill.textContent = 'Solicitando acesso à câmera...';
  statusPill.classList.remove('detected');
  statusPill.classList.add('not-detected');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user'
      },
      audio: false
    });

    console.log('[Anato3D] Stream da câmera obtida.', stream.getVideoTracks().length);

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    video.onloadedmetadata = async () => {
      console.log('[Anato3D] metadata do vídeo carregada.');
      resizeCanvas();

      try {
        await video.play();
        console.log('[Anato3D] Reprodução do vídeo iniciada.');
      } catch (playError) {
        console.warn('[Anato3D] Não foi possível iniciar a reprodução do vídeo:', playError);
      }

      try {
        await Promise.all([initPoseLandmarker(), initFaceLandmarker()]);
        startDetectionLoop();
      } catch (poseError) {
        console.error('[Anato3D] Erro ao iniciar a detecção corporal ou facial:', poseError);
        statusPill.textContent = 'Câmera ativada, mas a detecção corporal não pôde iniciar.';
      }
    };

    isCameraActive = true;
    startCameraBtn.textContent = 'CÂMERA ATIVA';
  } catch (error) {
    console.error('[Anato3D] Erro ao acessar a câmera:', error);
    updateStatus(false);
    statusPill.textContent = 'Não foi possível iniciar a câmera. Permita o acesso e tente novamente.';
    startCameraBtn.disabled = false;
    startCameraBtn.textContent = 'INICIAR CÂMERA';
  }
}

startCameraBtn.addEventListener('click', () => {
  startCamera();
});
setupThreeScene();
updateStatus(false);
