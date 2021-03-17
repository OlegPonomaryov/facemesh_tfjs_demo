const inputVideo = document.getElementById("inputVideo");
const outputCanvas = document.getElementById("outputCanvas");
const outputCanvasContext = outputCanvas.getContext("2d");


async function main() { 
    let net = await faceLandmarksDetection.load(
      faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
      {
        shouldLoadIrisModel: false,
        maxFaces: 1
      });
    
    const webcam = await tf.data.webcam(inputVideo);

    let videoWidth = inputVideo.videoWidth,
        videoHeight = inputVideo.videoHeight;
    outputCanvas.width = videoWidth;
    outputCanvas.height = videoHeight;

    outputCanvasContext.font = "18px Arial MS";

    let fps_ema = -1,
        prev_frame_time = -1;
    while (true) {
      const img = await webcam.capture();
      const predictions = await net.estimateFaces({
        input: img,
        predictIrises: false
      });

      outputCanvasContext.drawImage(inputVideo, 0, 0);
      plot_landmarks(predictions);

      let curr_frame_time = Date.now();
      if (prev_frame_time >= 0) {
        fps_ema = calc_fps(prev_frame_time, curr_frame_time, fps_ema);
      }
      outputCanvasContext.fillStyle = "red";
      outputCanvasContext.fillText(Math.round(fps_ema) + " FPS", 5, 20);
      prev_frame_time = curr_frame_time;

      img.dispose();
  
      await tf.nextFrame();
    }
}

function plot_landmarks(predictions) {
  outputCanvasContext.fillStyle = "green";
  if (predictions.length > 0) {
    for (let i = 0; i < predictions.length; i++) {
      const keypoints = predictions[i].scaledMesh;
      for (let i = 0; i < keypoints.length; i++) {
        const [x, y, z] = keypoints[i];
        
        outputCanvasContext.beginPath();
        outputCanvasContext.arc(x, y, 2, 0, 2 * Math.PI);
        outputCanvasContext.fill();
      }
    }
  }
}

function calc_fps(prev_frame_time, curr_frame_time, fps_ema) {
    let curr_fps = 1000 / (curr_frame_time - prev_frame_time);
    if (fps_ema >= 0)
    {
        fps_ema = 0.05 * curr_fps + (1 - 0.05) * fps_ema;
    }
    else
    {
        fps_ema = curr_fps;
    }
    return fps_ema;
}

main();
