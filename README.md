# CIFAR-10 Image Classifier (Streamlit App)

A portfolio-ready deep learning project: train a CNN on the CIFAR-10
dataset, then deploy it as a web app where anyone can upload a photo
and instantly see which of 10 categories the model thinks it belongs
to — **airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
truck**.

## Why this is a strong resume project
- Covers the full ML lifecycle: data prep → CNN training → evaluation
  → deployment.
- Deployed as a live, shareable link (not just a notebook).
- Interactive: recruiters/interviewers can actually upload a photo and
  see it work in real time.

## Project structure
```
cifar10-streamlit-app/
├── train_model.py      # trains the CNN, saves cifar10_model.h5
├── app.py               # Streamlit app: upload image -> predicted class
├── requirements.txt      # dependencies for local + cloud deployment
└── README.md
```

## 1. Train the model
```bash
pip install tensorflow numpy
python train_model.py
```
This downloads CIFAR-10 automatically, trains for ~30 epochs
(typically 75-85% test accuracy with this architecture), and saves
`cifar10_model.h5` in the same folder. On a laptop CPU this can take
30-60+ minutes; a free Google Colab GPU runtime finishes it in a few
minutes — train there, then download `cifar10_model.h5` into this
folder.

## 2. Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`)
and upload any image to test it.

## 3. Deploy for free (Streamlit Community Cloud)
1. Create a public GitHub repo and push this whole folder, including
   the trained `cifar10_model.h5`.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select your repo/branch, set:
   - Main file path: `app.py`
4. Click **Deploy**. In a minute or two you'll get a public link like:
   `https://your-app-name.streamlit.app`
5. Share that link on your resume/LinkedIn/GitHub README.

> `cifar10_model.h5` is usually a few MB, well within GitHub's normal
> file size limits, so no special storage is needed.

## Ideas to make it stand out further
- Add a "confusion matrix" and training-curve plot to the README as
  proof of model performance.
- Swap the custom CNN for transfer learning (MobileNetV2 fine-tuned on
  CIFAR-10) to push accuracy higher and mention "transfer learning" on
  your resume.
- Add a short screen recording/GIF of the app in the GitHub README.
