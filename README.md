# Sami Face Detector

A web application that finds Sami's face in a photo. You upload a picture, and a yellow box with a confidence score is drawn around the face it recognises.

**Live site (Vercel):** https://sami-face-detection.vercel.app

**Repository:** https://github.com/SamuelNartey-KO/sami-face-detection

## About the project

- **Dataset:** 96 photos of Sami, alone and in groups, labelled by hand on the Ultralytics Platform (67 for training, 14 for validation, 15 for testing). The photos are private and are not included in this repository.
- **Classes:** `Sammy` (Sami's face) and `Other` (other people's faces in the same photos). The website only draws boxes for `Sammy`.
- **Model:** YOLO26n (Ultralytics), trained for 100 epochs at an image size of 640 on a cloud GPU on the Ultralytics Platform
- **Export:** ONNX, full precision, so it runs in the browser
- **Website:** hand-built with HTML, CSS and JavaScript. The model runs in the browser with ONNX Runtime Web, so photos never leave the user's device.
- **Hosting:** Vercel, deployed from this repository

## Model results (validation set)

| Metric    | Score |
|-----------|-------|
| mAP50     | 67.4% |
| mAP50-95  | 45.9% |
| Precision | 61.0% |
| Recall    | 89.3% |

The model finds Sami's face reliably, and it sometimes also marks other faces. The page has a confidence slider and a "Show best match only" option to narrow the result down.

## Repository structure

- `model/` - the trained model (`sami-face-v2.onnx`)
- `index.html`, `style.css`, `script.js` - the website

## Run it locally

Serve the folder with a local server (for example VS Code Live Server, or `python -m http.server 8000`) and open `index.html`. Opening the file directly will not work, because browsers block reading the model file.

## Author

Samuel Nartey