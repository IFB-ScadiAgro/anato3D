import { FilesetResolver, PoseLandmarker, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest';
import { MTLLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/loaders/OBJLoader.js';

const video = document.getElementById('video');
const poseCanvas = document.getElementById('poseCanvas');
const statusPill = document.getElementById('statusPill');
const startCameraBtn = document.getElementById('startCameraBtn');
const toggleCameraBtn = document.getElementById('toggleCameraBtn');
const sceneContainer = document.getElementById('sceneContainer');
const fpsBadge = document.getElementById('fpsBadge');

const toggleBrain = document.getElementById('toggleBrain');
const toggleHeart = document.getElementById('toggleHeart');
const toggleLungs = document.getElementById('toggleLungs');
const toggleSkeleton = document.getElementById('toggleSkeleton');
const toggleVitalMotion = document.getElementById('toggleVitalMotion');

const badgeBrain = document.getElementById('badgeBrain');
const badgeHeart = document.getElementById('badgeHeart');
const badgeLungs = document.getElementById('badgeLungs');

const opacitySlider = document.getElementById('opacitySlider');
const opacityValue = document.getElementById('opacityValue');

const organCards = document.querySelectorAll('.organ-card');
const detailTitle = document.getElementById('detailTitle');
const detailDesc = document.getElementById('detailDesc');
const detailRegion = document.getElementById('detailRegion');
const detailRef = document.getElementById('detailRef');

const ctx = poseCanvas.getContext('2d');

const ORGAN_INFO = {
  brain: {
    title: 'Cérebro',
    desc: 'Principal órgão do sistema nervoso central, localizado na caixa craniana. Controla funções cognitivas, sensoriais, memória e comando motor.',
    region: 'Região: Cavidade Craniana',
    ref: 'Ancoragem: Malha Facial / Orbital 3D'
  },
  heart: {
    title: 'Coração',
    desc: 'Órgão muscular oco localizado no mediastino médio, levemente inclinado à esquerda. Responsável por bombear o sangue por todo o sistema circulatório.',
    region: 'Região: Mediastino / Caixa Torácica',
    ref: 'Ancoragem: Eixo Escapular (Ombros e Tronco)'
  },
  lungs: {
    title: 'Pulmões',
    desc: 'Órgãos esponjosos do sistema respiratório situados na cavidade torácica, protegidos pelas costelas. Realizam a hematose (troca gasosa de oxigênio e CO₂).',
    region: 'Região: Cavidade Pleural / Tórax',
    ref: 'Ancoragem: Eixo Escapular e Costal'
  }
};

const FACE_OUTLINE = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
  377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
  67, 109, 104, 71, 63, 105, 66, 107, 55, 65, 52, 53, 46
];

const FACE_GUIDE_POINTS = [1, 33, 61, 291, 199, 152, 263, 454, 234];

// Clean anatomical torso & limb connections (no crossed chest lines or noisy face-to-shoulder lines)
const BODY_POSE_CONNECTIONS = [
  [11, 12], // Collarbone / Shoulders
  [11, 13], [13, 15], // Left Arm
  [12, 14], [14, 16], // Right Arm
  [11, 23], [12, 24], // Torso sides
  [23, 24], // Pelvis / Hips
  [23, 25], [25, 27], // Left Leg
  [24, 26], [26, 28]  // Right Leg
];

const BODY_JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

let poseLandmarker = null;
let faceLandmarker = null;
let animationFrameId = null;
let isCameraActive = false;
let isCameraPaused = false;
let lastVideoTime = -1;

let threeScene = null;
let camera = null;
let renderer = null;
let headGroup = null;
let brainMesh = null;
let torsoGroup = null;
let heartMesh = null;
let lungsMesh = null;

let headPose = {
  x: 0,
  y: 0,
  scale: 1,
  yaw: 0,
  pitch: 0,
  roll: 0,
  initialized: false
};

let torsoPose = {
  x: 0,
  y: 0,
  scale: 1,
  yaw: 0,
  pitch: 0,
  roll: 0,
  torsoRatio: 1.25,
  initialized: false
};

let smoothedPoseLandmarks = null;
let organOpacity = 1.0;

const BRAIN_MODEL_BASE_URL = './assets/brain/';
const HEART_MODEL_BASE_URL = './assets/heart/';
const LUNGS_MODEL_BASE_URL = './assets/lungs/';

const organToggles = {
  brain: true,
  heart: true,
  lungs: true,
  skeleton: true,
  vitalMotion: true
};

// Frame rate tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();

function lerpWithDeadzone(current, target, alpha, deadzone = 0.002) {
  if (Math.abs(target - current) < deadzone) {
    return current;
  }
  return THREE.MathUtils.lerp(current, target, alpha);
}

function smoothHeadPose(target) {
  if (!headPose.initialized) {
    headPose.x = target.x;
    headPose.y = target.y;
    headPose.scale = target.scale;
    headPose.yaw = target.yaw;
    headPose.pitch = target.pitch;
    headPose.roll = target.roll;
    headPose.initialized = true;
    return;
  }

  headPose.x = lerpWithDeadzone(headPose.x, target.x, 0.22, 0.003);
  headPose.y = lerpWithDeadzone(headPose.y, target.y, 0.22, 0.003);
  headPose.scale = lerpWithDeadzone(headPose.scale, target.scale, 0.16, 0.004);
  headPose.yaw = lerpWithDeadzone(headPose.yaw, target.yaw, 0.15, 0.005);
  headPose.pitch = lerpWithDeadzone(headPose.pitch, target.pitch, 0.15, 0.005);
  headPose.roll = lerpWithDeadzone(headPose.roll, target.roll, 0.15, 0.005);
}

function smoothTorsoPose(target) {
  if (!torsoPose.initialized) {
    torsoPose.x = target.x;
    torsoPose.y = target.y;
    torsoPose.scale = target.scale;
    torsoPose.yaw = target.yaw;
    torsoPose.pitch = target.pitch;
    torsoPose.roll = target.roll;
    torsoPose.initialized = true;
    return;
  }

  torsoPose.x = lerpWithDeadzone(torsoPose.x, target.x, 0.18, 0.004);
  torsoPose.y = lerpWithDeadzone(torsoPose.y, target.y, 0.18, 0.004);
  torsoPose.scale = lerpWithDeadzone(torsoPose.scale, target.scale, 0.14, 0.005);
  torsoPose.yaw = lerpWithDeadzone(torsoPose.yaw, target.yaw, 0.1, 0.008);
  torsoPose.pitch = lerpWithDeadzone(torsoPose.pitch, target.pitch, 0.1, 0.008);
  torsoPose.roll = lerpWithDeadzone(torsoPose.roll, target.roll, 0.14, 0.005);
}

function isLandmarkConfident(point, minVisibility = 0.5, margin = 0.02) {
  if (!point) {
    return false;
  }
  const vis = point.visibility ?? 1;
  if (vis < minVisibility) {
    return false;
  }
  if (point.x < margin || point.x > 1 - margin || point.y < margin || point.y > 1 - margin) {
    return false;
  }
  return true;
}

function getSmoothedPoseLandmarks(rawLandmarks) {
  if (!rawLandmarks) {
    smoothedPoseLandmarks = null;
    return null;
  }

  if (!smoothedPoseLandmarks || smoothedPoseLandmarks.length !== rawLandmarks.length) {
    smoothedPoseLandmarks = rawLandmarks.map(pt => ({ ...pt }));
    return smoothedPoseLandmarks;
  }

  for (let i = 0; i < rawLandmarks.length; i++) {
    const raw = rawLandmarks[i];
    const prev = smoothedPoseLandmarks[i];
    if (!raw) {
      continue;
    }

    prev.x = lerpWithDeadzone(prev.x, raw.x, 0.35, 0.0015);
    prev.y = lerpWithDeadzone(prev.y, raw.y, 0.35, 0.0015);
    prev.z = lerpWithDeadzone(prev.z || 0, raw.z || 0, 0.2, 0.003);
    prev.visibility = THREE.MathUtils.lerp(prev.visibility ?? 1, raw.visibility ?? 1, 0.3);
  }

  return smoothedPoseLandmarks;
}

function applyMaterialOpacity(rootObject, opacity) {
  if (!rootObject) {
    return;
  }

  rootObject.traverse((child) => {
    if (!child.isMesh || !child.material) {
      return;
    }

    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((mat) => {
      mat.transparent = opacity < 0.99;
      mat.opacity = opacity;
      mat.needsUpdate = true;
    });
  });
}

function updateAllOrganOpacities() {
  applyMaterialOpacity(brainMesh, organOpacity);
  applyMaterialOpacity(heartMesh, organOpacity);
  applyMaterialOpacity(lungsMesh, organOpacity);
  if (renderer && threeScene && camera) {
    renderer.render(threeScene, camera);
  }
}

function applyOrganToggles() {
  if (brainMesh) {
    brainMesh.visible = organToggles.brain;
  }
  if (heartMesh) {
    heartMesh.visible = organToggles.heart;
  }
  if (lungsMesh) {
    lungsMesh.visible = organToggles.lungs;
  }
  if (renderer && threeScene && camera) {
    renderer.render(threeScene, camera);
  }
}

function updateBadgeState(badgeEl, isEnabled, isTracking) {
  if (!badgeEl) {
    return;
  }

  badgeEl.classList.remove('tracking', 'waiting', 'hidden-organ');
  if (!isEnabled) {
    badgeEl.textContent = 'Oculto';
    badgeEl.classList.add('hidden-organ');
  } else if (isTracking) {
    badgeEl.textContent = 'Rastreando';
    badgeEl.classList.add('tracking');
  } else {
    badgeEl.textContent = 'Fora de quadro';
    badgeEl.classList.add('waiting');
  }
}

function selectOrganDetail(organKey) {
  const info = ORGAN_INFO[organKey];
  if (!info) {
    return;
  }

  organCards.forEach((card) => {
    card.classList.toggle('active', card.dataset.organ === organKey);
  });

  if (detailTitle) detailTitle.textContent = info.title;
  if (detailDesc) detailDesc.textContent = info.desc;
  if (detailRegion) detailRegion.textContent = info.region;
  if (detailRef) detailRef.textContent = info.ref;
}

function setupUIListeners() {
  if (toggleBrain) {
    organToggles.brain = toggleBrain.checked;
    toggleBrain.addEventListener('change', (event) => {
      organToggles.brain = event.target.checked;
      applyOrganToggles();
    });
  }

  if (toggleHeart) {
    organToggles.heart = toggleHeart.checked;
    toggleHeart.addEventListener('change', (event) => {
      organToggles.heart = event.target.checked;
      applyOrganToggles();
    });
  }

  if (toggleLungs) {
    organToggles.lungs = toggleLungs.checked;
    toggleLungs.addEventListener('change', (event) => {
      organToggles.lungs = event.target.checked;
      applyOrganToggles();
    });
  }

  if (toggleSkeleton) {
    organToggles.skeleton = toggleSkeleton.checked;
    toggleSkeleton.addEventListener('change', (event) => {
      organToggles.skeleton = event.target.checked;
    });
  }

  if (toggleVitalMotion) {
    organToggles.vitalMotion = toggleVitalMotion.checked;
    toggleVitalMotion.addEventListener('change', (event) => {
      organToggles.vitalMotion = event.target.checked;
      if (!organToggles.vitalMotion) {
        if (heartMesh) heartMesh.scale.setScalar(1);
        if (lungsMesh) lungsMesh.scale.set(1, 1, 1);
      }
    });
  }

  if (opacitySlider && opacityValue) {
    opacitySlider.addEventListener('input', (event) => {
      const val = Number(event.target.value);
      opacityValue.textContent = `${val}%`;
      organOpacity = val / 100;
      updateAllOrganOpacities();
    });
  }

  organCards.forEach((card) => {
    card.addEventListener('click', () => {
      const organKey = card.dataset.organ;
      if (organKey) {
        selectOrganDetail(organKey);
      }
    });
  });

  if (toggleCameraBtn) {
    toggleCameraBtn.addEventListener('click', () => {
      if (!isCameraActive) {
        return;
      }
      isCameraPaused = !isCameraPaused;
      if (isCameraPaused) {
        video.pause();
        toggleCameraBtn.textContent = 'RETOMAR CÂMERA';
        statusPill.textContent = 'Câmera pausada';
      } else {
        video.play();
        toggleCameraBtn.textContent = 'PAUSAR CÂMERA';
      }
    });
  }
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

  const ambientLight = new THREE.AmbientLight(0xffffff, 1.65);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.35);
  directionalLight.position.set(2, 3, 5);

  const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.5);
  rimLight.position.set(-3, -2, 2);

  threeScene.add(ambientLight);
  threeScene.add(directionalLight);
  threeScene.add(rimLight);

  headGroup = new THREE.Group();

  const brainMaterial = new THREE.MeshStandardMaterial({
    color: 0x7dd3fc,
    emissive: 0x1d4ed8,
    transparent: true,
    opacity: 0.82,
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

  headGroup.add(brainMesh);
  headGroup.visible = false;
  threeScene.add(headGroup);

  torsoGroup = new THREE.Group();

  const lungsMaterial = new THREE.MeshStandardMaterial({
    color: 0xfda4af,
    emissive: 0x9f1239,
    transparent: true,
    opacity: 0.82,
    roughness: 0.5,
    metalness: 0.15
  });

  lungsMesh = new THREE.Group();
  const fallbackLeftLung = new THREE.Mesh(
    new THREE.SphereGeometry(0.45, 24, 24),
    lungsMaterial
  );
  fallbackLeftLung.scale.set(0.75, 1.35, 0.65);
  fallbackLeftLung.position.set(-0.42, 0, 0);

  const fallbackRightLung = new THREE.Mesh(
    new THREE.SphereGeometry(0.45, 24, 24),
    lungsMaterial
  );
  fallbackRightLung.scale.set(0.75, 1.35, 0.65);
  fallbackRightLung.position.set(0.42, 0, 0);

  lungsMesh.add(fallbackLeftLung);
  lungsMesh.add(fallbackRightLung);
  lungsMesh.position.set(0, 0.05, 0.15);

  const heartMaterial = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0x7f1d1d,
    transparent: true,
    opacity: 0.88,
    roughness: 0.4,
    metalness: 0.2
  });

  heartMesh = new THREE.Group();
  const fallbackHeart = new THREE.Mesh(
    new THREE.SphereGeometry(0.28, 24, 24),
    heartMaterial
  );
  fallbackHeart.scale.set(0.9, 1.15, 0.85);
  heartMesh.add(fallbackHeart);
  heartMesh.position.set(0.065, -0.075, 0.34);
  heartMesh.rotation.set(0.04, -0.08, 0.1);

  torsoGroup.add(lungsMesh);
  torsoGroup.add(heartMesh);
  torsoGroup.visible = false;

  threeScene.add(torsoGroup);

  loadBrainModel();
  loadLungsModel();
  loadHeartModel();
  applyOrganToggles();

  window.addEventListener('resize', () => {
    if (!sceneContainer) {
      return;
    }

    camera.aspect = sceneContainer.clientWidth / sceneContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
  });
}

async function loadObjWithMtl(baseUrl, mtlFile, objFile, targetSize) {
  const materialsLoader = new MTLLoader();
  const objectLoader = new OBJLoader();

  materialsLoader.setPath(baseUrl);
  const materials = await materialsLoader.loadAsync(mtlFile);
  materials.preload();

  objectLoader.setMaterials(materials);
  objectLoader.setPath(baseUrl);
  const model = await objectLoader.loadAsync(objFile);
  const bounds = new THREE.Box3().setFromObject(model);
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const largestDimension = Math.max(size.x, size.y, size.z) || 1;
  const modelScale = targetSize / largestDimension;

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

  return model;
}

async function loadBrainModel() {
  if (!brainMesh) {
    return;
  }

  try {
    const model = await loadObjWithMtl(
      BRAIN_MODEL_BASE_URL,
      'Meshy_AI_Human_Brain_Anatomy_0921190125_texture.mtl',
      'Meshy_AI_Human_Brain_Anatomy_0921190125_texture.obj',
      0.95
    );
    brainMesh.clear();
    brainMesh.add(model);
    applyMaterialOpacity(brainMesh, organOpacity);
    console.log('[Anato3D] Modelo OBJ do cérebro carregado com MTL e textura.');
  } catch (error) {
    console.warn('[Anato3D] Não foi possível carregar o cérebro OBJ. Usando fallback.', error);
  }
}

async function loadHeartModel() {
  if (!heartMesh) {
    return;
  }

  try {
    const model = await loadObjWithMtl(
      HEART_MODEL_BASE_URL,
      'Meshy_AI_Anatomical_Heart_0921131925_texture.mtl',
      'Meshy_AI_Anatomical_Heart_0921131925_texture.obj',
      0.56
    );
    heartMesh.clear();
    heartMesh.add(model);
    applyMaterialOpacity(heartMesh, organOpacity);
    console.log('[Anato3D] Modelo OBJ do coração carregado com MTL e textura.');
  } catch (error) {
    console.warn('[Anato3D] Não foi possível carregar o coração OBJ. Usando fallback.', error);
  }
}

async function loadLungsModel() {
  if (!lungsMesh) {
    return;
  }

  try {
    const model = await loadObjWithMtl(
      LUNGS_MODEL_BASE_URL,
      'Meshy_AI_Human_Lungs_0921185400_texture.mtl',
      'Meshy_AI_Human_Lungs_0921185400_texture.obj',
      1.50
    );
    lungsMesh.clear();
    lungsMesh.add(model);
    applyMaterialOpacity(lungsMesh, organOpacity);
    console.log('[Anato3D] Modelo OBJ dos pulmões carregado com MTL e textura.');
  } catch (error) {
    console.warn('[Anato3D] Não foi possível carregar os pulmões OBJ. Usando fallback.', error);
  }
}

function updateHeadFromFace(faceLandmarks) {
  if (!headGroup || !brainMesh) {
    return false;
  }

  const leftEye = faceLandmarks[33] || faceLandmarks[133];
  const rightEye = faceLandmarks[263] || faceLandmarks[362];
  const nose = faceLandmarks[1] || faceLandmarks[168];
  const chin = faceLandmarks[152] || faceLandmarks[14];
  const forehead = faceLandmarks[10] || faceLandmarks[151];

  if (
    !leftEye || !rightEye || !nose || !chin || !forehead ||
    forehead.y < 0.01 || nose.y < 0.02
  ) {
    headGroup.visible = false;
    headPose.initialized = false;
    return false;
  }

  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const eyeCenterY = (leftEye.y + rightEye.y) / 2;
  const eyeDistance = Math.max(0.06, Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y));
  const faceHeight = Math.max(0.12, chin.y - forehead.y);

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

  const targetYaw = THREE.MathUtils.clamp(-yaw * 1.15, -1.15, 1.15);
  const targetPitch = THREE.MathUtils.clamp(pitch, -0.8, 0.8);
  const targetRoll = THREE.MathUtils.clamp(roll * 1.4, -1.1, 1.1);

  smoothHeadPose({ x, y, scale, yaw: targetYaw, pitch: targetPitch, roll: targetRoll });

  headGroup.visible = organToggles.brain;
  headGroup.position.set(headPose.x, headPose.y, 0.8);
  headGroup.rotation.set(headPose.pitch, headPose.yaw, headPose.roll);
  headGroup.scale.setScalar(headPose.scale);

  return true;
}

function updateHeadFromPose(poseLandmarks) {
  if (!headGroup || !brainMesh || !poseLandmarks) {
    return false;
  }

  const nose = poseLandmarks[0];
  const leftEye = poseLandmarks[2];
  const rightEye = poseLandmarks[5];
  const leftShoulder = poseLandmarks[11];
  const rightShoulder = poseLandmarks[12];

  // Strict confidence & frame boundary check so we don't render a "phantom" head on the neck when off-screen
  if (
    !isLandmarkConfident(nose, 0.72, 0.04) ||
    !isLandmarkConfident(leftEye, 0.72, 0.04) ||
    !isLandmarkConfident(rightEye, 0.72, 0.04)
  ) {
    headGroup.visible = false;
    headPose.initialized = false;
    return false;
  }

  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const eyeCenterY = (leftEye.y + rightEye.y) / 2;
  const eyeDistance = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);

  // Reject collapsed/clustered head points (which happen when head is cut off at top of camera)
  if (eyeDistance < 0.035) {
    headGroup.visible = false;
    headPose.initialized = false;
    return false;
  }

  if (leftShoulder && rightShoulder) {
    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    const shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
    // Ensure head is anatomically well above shoulders
    if (shoulderMidY - nose.y < shoulderWidth * 0.28) {
      headGroup.visible = false;
      headPose.initialized = false;
      return false;
    }
  }

  const yaw = THREE.MathUtils.clamp((nose.x - eyeCenterX) / eyeDistance * 1.5, -1, 1);
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  const brainCenterY = eyeCenterY - (eyeDistance * 0.45);
  const mirroredCenterX = 1 - eyeCenterX;
  const x = (mirroredCenterX - 0.5) * 5.5;
  const y = (0.5 - brainCenterY) * 4.8;
  const skullWidth = eyeDistance * 2.55;
  const scale = THREE.MathUtils.clamp(skullWidth * 2.7, 0.68, 1.35);

  const targetYaw = THREE.MathUtils.clamp(-yaw * 1.15, -1.15, 1.15);
  const targetRoll = THREE.MathUtils.clamp(roll * 1.2, -1.1, 1.1);

  smoothHeadPose({ x, y, scale, yaw: targetYaw, pitch: 0, roll: targetRoll });

  headGroup.visible = organToggles.brain;
  headGroup.position.set(headPose.x, headPose.y, 0.8);
  headGroup.rotation.set(headPose.pitch, headPose.yaw, headPose.roll);
  headGroup.scale.setScalar(headPose.scale);

  return true;
}

function updateTorsoFromPose(poseLandmarks) {
  if (!torsoGroup || !poseLandmarks) {
    return false;
  }

  const leftShoulder = poseLandmarks[11];
  const rightShoulder = poseLandmarks[12];
  const leftHip = poseLandmarks[23];
  const rightHip = poseLandmarks[24];

  if (!isLandmarkConfident(leftShoulder, 0.45, 0.01) || !isLandmarkConfident(rightShoulder, 0.45, 0.01)) {
    torsoGroup.visible = false;
    torsoPose.initialized = false;
    return false;
  }

  const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
  const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
  const shoulderWidth = Math.max(
    0.08,
    Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y)
  );

  // Smoothly update torso ratio only when hips are clearly visible inside the frame
  const hipsConfident =
    isLandmarkConfident(leftHip, 0.6, 0.03) &&
    isLandmarkConfident(rightHip, 0.6, 0.03);

  let chestCenterX = shoulderMidX;
  if (hipsConfident) {
    const hipMidX = (leftHip.x + rightHip.x) / 2;
    const hipMidY = (leftHip.y + rightHip.y) / 2;
    const rawRatio = THREE.MathUtils.clamp((hipMidY - shoulderMidY) / shoulderWidth, 0.95, 1.65);
    torsoPose.torsoRatio = THREE.MathUtils.lerp(torsoPose.torsoRatio, rawRatio, 0.08);
    chestCenterX = THREE.MathUtils.lerp(shoulderMidX, hipMidX, 0.22);
  }

  // Anchor chest Y primarily to shoulders + smoothed torso ratio so hip jitter never causes vertical jumps
  const chestCenterY = shoulderMidY + (shoulderWidth * torsoPose.torsoRatio * 0.34);

  const mirroredCenterX = 1 - chestCenterX;
  const x = (mirroredCenterX - 0.5) * 5.5;
  const y = (0.5 - chestCenterY) * 4.8;

  const scale = THREE.MathUtils.clamp(shoulderWidth * 2.82, 0.55, 1.85);

  const dxScreen = (1 - leftShoulder.x) - (1 - rightShoulder.x);
  const dyScreen = leftShoulder.y - rightShoulder.y;
  const roll = THREE.MathUtils.clamp(Math.atan2(-dyScreen, Math.max(0.06, Math.abs(dxScreen))), -0.65, 0.65);

  const zDiff = (rightShoulder.z || 0) - (leftShoulder.z || 0);
  const yaw = THREE.MathUtils.clamp((zDiff / shoulderWidth) * 0.75, -0.75, 0.75);

  smoothTorsoPose({ x, y, scale, yaw, pitch: 0, roll });

  torsoGroup.visible = organToggles.heart || organToggles.lungs;
  torsoGroup.position.set(torsoPose.x, torsoPose.y, 0.6);
  torsoGroup.rotation.set(torsoPose.pitch, torsoPose.yaw, torsoPose.roll);
  torsoGroup.scale.setScalar(torsoPose.scale);

  if (heartMesh) {
    heartMesh.visible = organToggles.heart;
  }
  if (lungsMesh) {
    lungsMesh.visible = organToggles.lungs;
  }

  return true;
}

function updateVitalAnimations(timestampMs) {
  if (!organToggles.vitalMotion) {
    return;
  }

  const t = timestampMs * 0.001;

  // Realistic double-pulse heartbeat (~72 BPM)
  if (heartMesh && torsoGroup && torsoGroup.visible && organToggles.heart) {
    const cycle = (t * 1.2) % 1;
    const pulse1 = Math.exp(-Math.pow((cycle - 0.15) / 0.055, 2)) * 0.07;
    const pulse2 = Math.exp(-Math.pow((cycle - 0.30) / 0.07, 2)) * 0.045;
    const heartBeatScale = 1 + pulse1 + pulse2;
    heartMesh.scale.setScalar(heartBeatScale);
  }

  // Gentle respiratory expansion (~15 breaths/min)
  if (lungsMesh && torsoGroup && torsoGroup.visible && organToggles.lungs) {
    const breath = (Math.sin(t * 1.55) + 1) * 0.5;
    lungsMesh.scale.set(
      1 + breath * 0.038,
      1 + breath * 0.025,
      1 + breath * 0.045
    );
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
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
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
      minFaceDetectionConfidence: 0.55,
      minFacePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
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
  if (!organToggles.skeleton || !landmarks) {
    return;
  }

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // Draw clean anatomical connections only when both endpoints are confident and inside frame
  for (const [startIndex, endIndex] of BODY_POSE_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];

    if (!isLandmarkConfident(start, 0.45, 0.015) || !isLandmarkConfident(end, 0.45, 0.015)) {
      continue;
    }

    const startX = (1 - start.x) * poseCanvas.width;
    const startY = start.y * poseCanvas.height;
    const endX = (1 - end.x) * poseCanvas.width;
    const endY = end.y * poseCanvas.height;

    const alpha = Math.min(start.visibility ?? 1, end.visibility ?? 1) * 0.55;

    ctx.lineWidth = 2.4;
    ctx.strokeStyle = `rgba(125, 211, 252, ${alpha.toFixed(2)})`;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  }

  // Draw only clean body joints (excluding noisy face/finger landmarks when out of frame)
  for (const index of BODY_JOINT_INDICES) {
    const point = landmarks[index];
    if (!isLandmarkConfident(point, 0.5, 0.015)) {
      continue;
    }

    const x = (1 - point.x) * poseCanvas.width;
    const y = point.y * poseCanvas.height;

    ctx.beginPath();
    ctx.fillStyle = 'rgba(94, 234, 212, 0.25)';
    ctx.arc(x, y, 6.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.fillStyle = '#e2f1ff';
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawFaceLandmarks(faceLandmarks) {
  if (!organToggles.skeleton || !faceLandmarks || faceLandmarks.length === 0) {
    return;
  }

  const forehead = faceLandmarks[10] || faceLandmarks[151];
  const nose = faceLandmarks[1] || faceLandmarks[168];
  if (!forehead || !nose || forehead.y < 0.01 || nose.y < 0.02) {
    return;
  }

  ctx.save();
  ctx.lineWidth = 1.8;
  ctx.strokeStyle = 'rgba(52, 211, 153, 0.65)';
  ctx.fillStyle = 'rgba(220, 252, 231, 0.85)';
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
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
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

function updateFpsCounter(now) {
  frameCount++;
  if (now - lastFpsUpdate >= 600) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
    if (fpsBadge) {
      fpsBadge.textContent = `${fps} FPS • 3D ATIVO`;
    }
    frameCount = 0;
    lastFpsUpdate = now;
  }
}

function processFrame() {
  if (!video || video.readyState < 2 || isCameraPaused) {
    return;
  }

  const now = performance.now();
  updateFpsCounter(now);

  // Only run MediaPipe inference when the video frame has updated
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    ctx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);

    let rawPoseLandmarks = null;
    let faceLandmarks = null;

    if (poseLandmarker) {
      const poseResult = poseLandmarker.detectForVideo(video, now);
      if (poseResult.landmarks && poseResult.landmarks.length > 0) {
        rawPoseLandmarks = poseResult.landmarks[0];
      }
    }

    if (faceLandmarker) {
      const faceResult = faceLandmarker.detectForVideo(video, now + 0.1);
      if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
        faceLandmarks = faceResult.faceLandmarks[0];
      }
    }

    const poseLandmarks = getSmoothedPoseLandmarks(rawPoseLandmarks);

    let headDetected = false;
    let torsoDetected = false;

    if (poseLandmarks) {
      drawSkeleton(poseLandmarks);
      torsoDetected = updateTorsoFromPose(poseLandmarks);
    } else if (torsoGroup) {
      torsoGroup.visible = false;
      torsoPose.initialized = false;
    }

    if (faceLandmarks) {
      drawFaceLandmarks(faceLandmarks);
      headDetected = updateHeadFromFace(faceLandmarks);
    } else if (poseLandmarks) {
      headDetected = updateHeadFromPose(poseLandmarks);
    } else if (headGroup) {
      headGroup.visible = false;
      headPose.initialized = false;
    }

    updateBadgeState(badgeBrain, organToggles.brain, headDetected);
    updateBadgeState(badgeHeart, organToggles.heart, torsoDetected);
    updateBadgeState(badgeLungs, organToggles.lungs, torsoDetected);

    if (headDetected || torsoDetected) {
      const detectedParts = [];
      if (headDetected) {
        const normalizedYaw = Math.abs(headPose.yaw);
        const orientationLabel = normalizedYaw < 0.3
          ? 'frontal'
          : normalizedYaw < 1.0
            ? 'lateral'
            : 'girando';
        detectedParts.push(`Cabeça (${orientationLabel})`);
      }
      if (torsoDetected) {
        detectedParts.push('Tórax (Coração e Pulmões)');
      }

      statusPill.textContent = `Rastreando: ${detectedParts.join(' + ')}`;
      statusPill.classList.remove('not-detected');
      statusPill.classList.add('detected');
    } else if (poseLandmarks) {
      updateStatus(true);
    } else {
      updateStatus(false);
    }
  }

  updateVitalAnimations(now);

  if (renderer && threeScene && camera) {
    renderer.render(threeScene, camera);
  }
}

function startDetectionLoop() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  const loop = () => {
    processFrame();
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
        if (toggleCameraBtn) {
          toggleCameraBtn.classList.remove('hidden');
        }
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
setupUIListeners();
setupThreeScene();
updateStatus(false);
