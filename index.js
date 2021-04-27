const FACE_DETECT_SIZE = 128;
const FACE_MESH_SIZE = 192;


const inputVideo = document.getElementById("inputVideo");
const outputCanvas = document.getElementById("outputCanvas");
const outputCanvasContext = outputCanvas.getContext("2d");
const backendOutput = document.getElementById("backendOutput");


async function main(blazefaceAnchors) {
  const settings = loadSettings();

  const blazefaceAnchorsTensor = tf.tensor(blazefaceAnchors);

  const faceDetModel = await tf.loadGraphModel("https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1", { fromTFHub: true });
  const faceMeshModel = await tf.loadGraphModel("https://tfhub.dev/mediapipe/tfjs-model/facemesh/1/default/1", { fromTFHub: true });

  const webcam = await tf.data.webcam(inputVideo);

  // Size of the source video
  const sourceSize = [inputVideo.videoHeight, inputVideo.videoWidth]
  outputCanvas.height = sourceSize[0];
  outputCanvas.width = sourceSize[1];

  // Size of the source video, resized to fit into 128x128 without changing its aspect ratio
  const faceDetSize = sourceSize[0] > sourceSize[1] ?
    [FACE_DETECT_SIZE, Math.round(sourceSize[1] * FACE_DETECT_SIZE / sourceSize[0])] :
    (sourceSize[1] > sourceSize[0] ?
      [Math.round(sourceSize[0] * FACE_DETECT_SIZE / sourceSize[1]), FACE_DETECT_SIZE] :
      [FACE_DETECT_SIZE, FACE_DETECT_SIZE]);

  // Padding of the resized video to make it exactly 128x128
  const faceDetPadding = [[Math.ceil((FACE_DETECT_SIZE - faceDetSize[0]) / 2), Math.floor((FACE_DETECT_SIZE - faceDetSize[0]) / 2)],
                          [Math.ceil((FACE_DETECT_SIZE - faceDetSize[1]) / 2), Math.floor((FACE_DETECT_SIZE - faceDetSize[1]) / 2)],
                          [0, 0]]

  outputCanvasContext.font = "18px Arial MS";

  let fpsEMA = -1,
      prevFrameTime = -1,
      approxFaceRect = null;
  while (true) {
    const img = await webcam.capture();

    const faceRect = approxFaceRect != null ? approxFaceRect :
      await runFaceDet(faceDetModel, img, sourceSize, faceDetSize, faceDetPadding, blazefaceAnchorsTensor);
    padRect(faceRect, sourceSize, 0.25);

    const faceMesh = faceRect[0] > 0.9 ? await runFaceMesh(faceMeshModel, img, faceRect) : null;
    approxFaceRect = faceMesh != null && faceMesh[0] > 0.5 ? faceMesh[2] : null;

    outputCanvasContext.drawImage(inputVideo, 0, 0);
    if (faceRect[0] > 0.9 && settings.get("boundingBox"))
      plotFaceRect(faceRect);
    if (faceMesh != null && faceMesh[0] > 0.5) {
      plotLandmarks(faceMesh[1]);
    }

    let currFrameTime = Date.now();
    if (prevFrameTime >= 0) {
      fpsEMA = calcFPS(prevFrameTime, currFrameTime, fpsEMA);
    }
    outputCanvasContext.fillStyle = "red";
    outputCanvasContext.fillText(Math.round(fpsEMA) + " FPS", 5, 20);
    prevFrameTime = currFrameTime;

    img.dispose();

    await tf.nextFrame();
  }
}


function loadSettings() {
  const url = new URL(window.location.href);

  const backend = url.searchParams.get("back") ?? "webgl";
  tf.setBackend(backend);
  backendOutput.innerText = "Backend: " + tf.getBackend();

  inputVideo.width = url.searchParams.get("width") ?? 640;
  inputVideo.height = url.searchParams.get("height") ?? 420;

  const boundingBox = url.searchParams.get("boundingBox") != null ?
    url.searchParams.get("boundingBox") == "true" : false;

  return new Map([
    ["boundingBox", boundingBox]
  ]); 
}


async function runFaceDet(faceDetModel, img, sourceSize, targetSize, padding, anchors) {
  const faceDetInput = tf.tidy(() => preprocessFaceDet(img, targetSize, padding));

  let predictions = await faceDetModel.predict(faceDetInput);
  faceDetInput.dispose();

  const result = tf.tidy(() => getBestRect(predictions, anchors));
  predictions.dispose();

  const faceRect = await result[0].data();
  result[0].dispose();
  const anchor = await result[1].data();
  result[1].dispose();

  postprocessFaceDet(faceRect, anchor, sourceSize, targetSize, padding);

  return faceRect;
}

// Transforms the input image into a (128 x 128) tensor while keeping the aspect
// ratio (what is expected by the corresponding face detection model), resulting
// in potential letterboxing in the transformed image.
function preprocessFaceDet(img, targetSize, padding) {
  const scale = tf.scalar(127.5);
  const offset = tf.scalar(1);
  const result = img.resizeBilinear(targetSize)
    .div(scale)
    .sub(offset)
    .pad(padding, 0)
    .reshape([1, FACE_DETECT_SIZE, FACE_DETECT_SIZE, 3]);
  return result;
}


function getBestRect(predictions, anchors) {
  const squeezedPred = predictions.squeeze();
  const logits = squeezedPred.slice([0, 0], [-1, 1]).squeeze();
  const bestRectIDX = logits.argMax();
  const bestPred = squeezedPred.gather(bestRectIDX).squeeze();
  const bestRect = bestPred.slice(0, 5);
  const anchor = anchors.gather(bestRectIDX).squeeze();
  return [bestRect, anchor];
}


function postprocessFaceDet(rect, anchor, sourceSize, targetSize, padding) {
  // Face probability is returned as logits, so sigmoid is applied
  // to get it in [0, 1] range
  rect[0] = 1 / (1 + Math.exp(-rect[0]));

  rect[1] += anchor[0] * FACE_DETECT_SIZE - padding[1][0];
  rect[2] += anchor[1] * FACE_DETECT_SIZE - padding[0][0];
  rect[3] *= anchor[2];
  rect[4] *= anchor[3];

  scale = sourceSize[0] / targetSize[0];
  for (let i = 1; i < 5; i++) {
    rect[i] *= scale;
  }

  rect[1] -= rect[3] / 2;
  rect[2] -= rect[4] / 2;
  rect[3] += rect[1];
  rect[4] += rect[2];
}


// Increases face bounding box size to make it bound the face less tight,
// which is required by the FaceMesh model
function padRect(rect, sourceSize, scale) {
  const widthPad = (rect[3] - rect[1]) * scale,
        heightPad = (rect[4] - rect[2]) * scale;

  rect[1] -= widthPad;
  rect[3] += widthPad;
  rect[2] -= heightPad;
  rect[4] += heightPad;

  for (let i = 1; i < 5; i++) {
    rect[i] = Math.round(rect[i]);
  }

  rect[1] = Math.max(0, rect[1]);
  rect[2] = Math.max(0, rect[2]);
  rect[3] = Math.min(sourceSize[1] - 1, rect[3]);
  rect[4] = Math.min(sourceSize[0] - 1, rect[4]);
}


async function runFaceMesh(faceMeshModel, img, faceRect) {
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


function plotFaceRect(faceRect) {
  outputCanvasContext.strokeStyle = "purple";
  outputCanvasContext.lineWidth = 2;
  outputCanvasContext.beginPath();
  outputCanvasContext.rect(
    faceRect[1], faceRect[2], faceRect[3] - faceRect[1], faceRect[4] - faceRect[2]);
  outputCanvasContext.stroke();
}


function calcFPS(prevFrameTime, currFrameTime, fpsEMA) {
  const currFPS = 1000 / (currFrameTime - prevFrameTime);
  if (fpsEMA >= 0) {
    const k = 0.01;
    fpsEMA = k * currFPS + (1 - k) * fpsEMA;
  }
  else {
    fpsEMA = currFPS;
  }
  return fpsEMA;
}
