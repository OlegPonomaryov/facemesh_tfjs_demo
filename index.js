const FACE_MESH_SIZE = 192;


const inputVideo = document.getElementById("inputVideo");
const outputCanvas = document.getElementById("outputCanvas");
const outputCanvasContext = outputCanvas.getContext("2d");
const backendOutput = document.getElementById("backendOutput");


async function main(blazefaceAnchors) {
  loadSettings();


  tf.enableProdMode();
  const blazefaceAnchorsTensor = tf.tensor(blazefaceAnchors);
  const faceMeshModel = await tf.loadGraphModel("https://tfhub.dev/mediapipe/tfjs-model/facemesh/1/default/1", { fromTFHub: true });

  const webcam = await tf.data.webcam(inputVideo);

  const sourceSize = [inputVideo.videoHeight, inputVideo.videoWidth]

  outputCanvas.height = sourceSize[0];
  outputCanvas.width = sourceSize[1];
  outputCanvasContext.font = "18px Arial MS";

  const faceRectSize = Math.min(...sourceSize) / 2;
  let faceRect = [1,
    Math.round(sourceSize[1] / 2 - faceRectSize / 2),
    Math.round(sourceSize[0] / 2 - faceRectSize / 2),
    Math.round(sourceSize[1] / 2 + faceRectSize / 2),
    Math.round(sourceSize[0] / 2 + faceRectSize / 2)];
  console.log(faceRect);


  let fpsEMA = -1,
    prevFrameTime = -1;
  while (true) {
    const img = await webcam.capture();

    faceMesh = await detectFaceMesh(faceMeshModel, img, faceRect);

    outputCanvasContext.drawImage(inputVideo, 0, 0);
    plotFaceRect(faceRect);
    if (faceMesh[0] > 0.5) {
      plotLandmarks(faceMesh[1]);
      let approxFaceRect = faceMesh[2];
      padRect(approxFaceRect, sourceSize, 0.25);
      faceRect = approxFaceRect;
    }

    let currFrameTime = Date.now();
    if (prevFrameTime >= 0) {
      fpsEMA = calcFPS(prevFrameTime, currFrameTime, fpsEMA);
    }
    outputCanvasContext.fillStyle = "red";
    outputCanvasContext.fillText(Math.round(fpsEMA) + " FPS", 5, 20);
    prevFrameTime = currFrameTime;

    img.dispose();
    console.log(tf.memory());

    await tf.nextFrame();
  }
}

function loadSettings() {
  let url = new URL(window.location.href);

  let backend = url.searchParams.get("back") ?? "webgl";
  tf.setBackend(backend);
  backendOutput.innerText = "Backend: " + tf.getBackend();
}

function plotFaceRect(faceRect) {
  outputCanvasContext.strokeStyle = "purple";
  outputCanvasContext.lineWidth = 2;
  outputCanvasContext.beginPath();
  outputCanvasContext.rect(
    faceRect[1], faceRect[2], faceRect[3] - faceRect[1], faceRect[4] - faceRect[2]);
  outputCanvasContext.stroke();
}

function padRect(rect, sourceSize, scale) {
  const widthPad = Math.round((rect[3] - rect[1]) * scale),
    heightPad = Math.round((rect[4] - rect[2]) * scale);
  for (let i = 1; i < 5; i++) {
    rect[i] = Math.round(rect[i]);
  }

  rect[1] -= widthPad;
  rect[3] += widthPad;
  rect[2] -= heightPad;
  rect[4] += heightPad;

  rect[1] = Math.max(0, rect[1]);
  rect[2] = Math.max(0, rect[2]);
  rect[3] = Math.min(sourceSize[1] - 1, rect[3]);
  rect[4] = Math.min(sourceSize[0] - 1, rect[4]);
}

async function detectFaceMesh(faceMeshModel, img, faceRect) {
  const faceMeshInput = tf.tidy(() => preprocessFaceMesh(img, faceRect));

  const predictions = await Promise.all(faceMeshModel.predict(faceMeshInput));
  faceMeshInput.dispose();

  const resultTensors = tf.tidy(() => postprocessFaceMesh(predictions, faceRect));

  const result = await Promise.all(resultTensors.map(async d => d.array()));
  for (let i = 0; i < resultTensors.length; i++) {
    resultTensors[i].dispose();
  }
  for (let i = 0; i < predictions.length; i++) {
    predictions[i].dispose();
  }

  const faceProb = result[0][0];
  const faceMesh = result[1];

  let approxFaceRect = [1];
  for (let i = 2; i < 6; i++) {
    approxFaceRect.push(result[i]);
  }

  return [faceProb, faceMesh, approxFaceRect];
}

function preprocessFaceMesh(img, faceRect) { 
  const scale = tf.scalar(255);
  const batchedImg = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
  const boxes = [[faceRect[2] / img.shape[0], faceRect[1] / img.shape[1],
                  faceRect[4] / img.shape[0], faceRect[3] / img.shape[1]]];
  const boxInd = [0];
  const cropSize = [FACE_MESH_SIZE, FACE_MESH_SIZE];
  const result = tf.image.cropAndResize(
    batchedImg, boxes, boxInd, cropSize).div(scale);
  return result;
}

function postprocessFaceMesh(predictions, faceRect) {
  const prob = predictions[1];
  const scale = tf.tensor([(faceRect[3] - faceRect[1]) / FACE_MESH_SIZE, (faceRect[4] - faceRect[2]) / FACE_MESH_SIZE, 1])
  const offset = tf.tensor([faceRect[1], faceRect[2], 0])
  const predMesh = predictions[2].reshape([-1, 3]);
  const mesh = predMesh.mul(scale).add(offset);
  const x = mesh.slice([0, 0], [-1, 1]);
  const y = mesh.slice([0, 1], [-1, 1]);
  return [prob, mesh, x.min(), y.min(), x.max(), y.max()];
}

function plotLandmarks(predictions) {
  outputCanvasContext.fillStyle = "green";
  for (let i = 0; i < predictions.length; i++) {
    outputCanvasContext.beginPath();
    outputCanvasContext.arc(Math.round(predictions[i][0]), Math.round(predictions[i][1]), 2, 0, 2 * Math.PI);
    outputCanvasContext.fill();
  }
}

function calcFPS(prevFrameTime, currFrameTime, fpsEMA) {
  let currFPS = 1000 / (currFrameTime - prevFrameTime);
  if (fpsEMA >= 0) {
    fpsEMA = 0.05 * currFPS + (1 - 0.05) * fpsEMA;
  }
  else {
    fpsEMA = currFPS;
  }
  return fpsEMA;
}
