<div align="center">

  <h1>🫁 Deep Learning-Based Pneumonia Detection</h1>
  <h3>ResNet18 • PyTorch • Flask • Docker • Render</h3>

  <p>
    An end-to-end deep learning project that detects <b>PNEUMONIA</b> or <b>NORMAL</b> from chest X-ray images,
    built with PyTorch, deployed with Flask, containerized using Docker, and hosted on Render.
  </p>

  <p>
    <a href="https://deep-learning-computer-vision-project.onrender.com/">
      <img src="https://img.shields.io/badge/Live%20Demo-Visit%20App-2563eb?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo">
    </a>
    <a href="https://github.com/">
      <img src="https://img.shields.io/badge/Status-Deployed-16a34a?style=for-the-badge" alt="Status">
    </a>
    <img src="https://img.shields.io/badge/Model-ResNet18-f97316?style=for-the-badge" alt="Model">
    <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </p>

  <p>
    <a href="https://deep-learning-computer-vision-project.onrender.com/"><b>🌐 Live Demo</b></a> •
    <a href="#-features"><b>Features</b></a> •
    <a href="#-tech-stack"><b>Tech Stack</b></a> •
    <a href="#-deployment"><b>Deployment</b></a> •
    <a href="#-project-structure"><b>Structure</b></a>
  </p>

</div>

<hr>

<h2>📌 Project Overview</h2>

<p>
This project is a <b>full-stack deep learning application</b> that classifies chest X-ray images into:
</p>

<ul>
  <li><b>NORMAL</b></li>
  <li><b>PNEUMONIA</b></li>
</ul>

<p>
It does not stop at model training. The project covers the complete workflow:
</p>

<ul>
  <li>dataset preparation and preprocessing</li>
  <li>deep learning model training using transfer learning</li>
  <li>validation and testing</li>
  <li>saving and loading trained weights</li>
  <li>building a Flask web application for inference</li>
  <li>containerizing the app with Docker</li>
  <li>deploying the app publicly on Render</li>
</ul>

<hr>

<h2 id="-features">🚀 Features</h2>

<table>
  <tr>
    <td>🧠</td>
    <td><b>Deep Learning Model</b></td>
    <td>Pretrained ResNet18 fine-tuned for pneumonia detection</td>
  </tr>
  <tr>
    <td>🖼️</td>
    <td><b>Image Classification</b></td>
    <td>Predicts whether a chest X-ray is NORMAL or PNEUMONIA</td>
  </tr>
  <tr>
    <td>📂</td>
    <td><b>Custom Dataset Pipeline</b></td>
    <td>PyTorch Dataset and DataLoader for train / val / test data</td>
  </tr>
  <tr>
    <td>🌐</td>
    <td><b>Flask Web App</b></td>
    <td>Upload image and get prediction in a browser</td>
  </tr>
  <tr>
    <td>🎨</td>
    <td><b>Modern Frontend</b></td>
    <td>Animated UI, clean layout, image preview, styled result feedback</td>
  </tr>
  <tr>
    <td>🐳</td>
    <td><b>Dockerized</b></td>
    <td>Packaged for consistent deployment and portability</td>
  </tr>
  <tr>
    <td>☁️</td>
    <td><b>Cloud Deployment</b></td>
    <td>Hosted on Render and accessible through a public live link</td>
  </tr>
</table>

<hr>

<h2>🧠 Model Details</h2>

<table>
  <tr>
    <th>Component</th>
    <th>Details</th>
  </tr>
  <tr>
    <td><b>Architecture</b></td>
    <td>ResNet18 (pretrained on ImageNet)</td>
  </tr>
  <tr>
    <td><b>Task</b></td>
    <td>Binary image classification</td>
  </tr>
  <tr>
    <td><b>Classes</b></td>
    <td>NORMAL / PNEUMONIA</td>
  </tr>
  <tr>
    <td><b>Framework</b></td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td><b>Loss Function</b></td>
    <td>CrossEntropyLoss</td>
  </tr>
  <tr>
    <td><b>Optimizer</b></td>
    <td>Adam</td>
  </tr>
  <tr>
    <td><b>Input Size</b></td>
    <td>224 × 224</td>
  </tr>
  <tr>
    <td><b>Training Environment</b></td>
    <td>Google Colab GPU (Tesla T4)</td>
  </tr>
</table>

<hr>

<h2>🗂️ Dataset</h2>

<p>
The project uses a chest X-ray image dataset organized into:
</p>

<pre>
train/
val/
test/
</pre>

<p>
Each split contains:
</p>

<pre>
NORMAL/
PNEUMONIA/
</pre>

<p>
Approximate split used in the project:
</p>

<ul>
  <li><b>Train:</b> 5216 images</li>
  <li><b>Validation:</b> 16 images</li>
  <li><b>Test:</b> 624 images</li>
</ul>

<hr>

<h2 id="-tech-stack">⚙️ Tech Stack</h2>

<h3>Machine Learning</h3>
<ul>
  <li>PyTorch</li>
  <li>Torchvision</li>
  <li>Pillow (PIL)</li>
  <li>Transfer Learning with ResNet18</li>
</ul>

<h3>Backend</h3>
<ul>
  <li>Flask</li>
  <li>Gunicorn</li>
</ul>

<h3>Frontend</h3>
<ul>
  <li>HTML5</li>
  <li>CSS3</li>
  <li>JavaScript</li>
</ul>

<h3>DevOps / Deployment</h3>
<ul>
  <li>Docker</li>
  <li>Render</li>
  <li>GitHub</li>
</ul>

<h3>Environment</h3>
<ul>
  <li>Google Colab</li>
  <li>Python 3</li>
</ul>

<hr>

<h2>🔁 End-to-End Workflow</h2>

<ol>
  <li>Mounted dataset in Google Colab</li>
  <li>Preprocessed chest X-ray images with transforms</li>
  <li>Built a custom <code>PneumoniaDataset</code> class using PyTorch</li>
  <li>Created train, validation, and test DataLoaders</li>
  <li>Loaded pretrained <b>ResNet18</b> and replaced final layer for 2-class output</li>
  <li>Trained the model using GPU acceleration</li>
  <li>Evaluated performance on validation and test sets</li>
  <li>Saved trained model weights as <code>pneumonia_classifier.pth</code></li>
  <li>Integrated the model into a Flask web application</li>
  <li>Built a polished frontend for image upload and result display</li>
  <li>Containerized the app using Docker</li>
  <li>Deployed the application on Render</li>
</ol>

<hr>

<h2>🌐 Web Application</h2>

<p>
The web application allows users to:
</p>

<ul>
  <li>upload a chest X-ray image</li>
  <li>run the trained model for prediction</li>
  <li>view the result directly in the browser</li>
</ul>

<p>
The deployed app is available here:
</p>

<p>
  <a href="https://deep-learning-computer-vision-project.onrender.com/">
    <b>👉 https://deep-learning-computer-vision-project.onrender.com/</b>
  </a>
</p>

<hr>

<h2>🐳 Docker</h2>

<p>
The application was containerized using Docker to ensure:
</p>

<ul>
  <li>consistent environment setup</li>
  <li>portable deployment</li>
  <li>easier cloud hosting</li>
</ul>

<p>
This made the project stronger from a software engineering and deployment perspective, not just a notebook-based ML experiment.
</p>

<hr>

<h2 id="-deployment">☁️ Deployment</h2>

<p>
The project is deployed on <b>Render</b> and accessible publicly through the live link above.
</p>

<p>
Deployment included:
</p>

<ul>
  <li>linking the GitHub repository</li>
  <li>configuring the web service</li>
  <li>running the Flask app in production</li>
  <li>handling model loading inside the deployed application</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
.
├── train_model.py
├── requirements.txt
├── Dockerfile
├── pneumonia_classifier.pth
├── .dockerignore
├── README.md
└── app/
    ├── app.py
    └── templates/
        └── index.html
</pre>

<hr>

<h2>📚 Key Learnings</h2>

<ul>
  <li>how CNN training works in a real project</li>
  <li>how to use transfer learning with pretrained models</li>
  <li>how to preprocess image datasets properly</li>
  <li>how to integrate ML models into a Flask app</li>
  <li>how to containerize and deploy an ML web app</li>
  <li>how real-world deployment issues differ from notebook experiments</li>
</ul>

<hr>

<h2>🔮 Future Improvements</h2>

<ul>
  <li>add confidence score with predictions</li>
  <li>add Grad-CAM for model explainability</li>
  <li>improve validation strategy with larger validation split</li>
  <li>add API endpoint for programmatic access</li>
  <li>optimize inference speed and model size</li>
</ul>

<hr>

<h2>👤 Author</h2>

<p>
<b>Neelmani Bhardwaj</b><br>
Computer Science Student • Backend & Machine Learning Enthusiast
</p>

<hr>

<div align="center">
  <h3>⭐ If you like the project, consider starring the repo.</h3>
</div>
