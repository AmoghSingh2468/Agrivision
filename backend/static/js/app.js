const API_URL = '/api';
let selectedFile = null;

const imageInput = document.getElementById('imageInput');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');

imageInput.addEventListener('change', handleFileSelect);
analyzeBtn.addEventListener('click', analyzeImage);

function handleFileSelect(event) {
    selectedFile = event.target.files[0];
    
    if (selectedFile) {
        console.log('File selected:', selectedFile.name);
        fileName.textContent = selectedFile.name;
        analyzeBtn.disabled = false;
    } else {
        fileName.textContent = 'No file chosen';
        analyzeBtn.disabled = true;
    }
}

async function analyzeImage() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    console.log('Analyzing image...');
    loadingSpinner.style.display = 'block';
    resultsSection.style.display = 'none';
    analyzeBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        console.log('Sending request to:', `${API_URL}/predict`);
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Prediction data:', data);
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}\n\nMake sure backend is running!`);
    } finally {
        loadingSpinner.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    document.getElementById('diseaseName').textContent = data.prediction.disease;
    document.getElementById('confidence').textContent = `Confidence: ${data.prediction.confidence}%`;
    
    document.getElementById('uploadedImage').src = data.images.original;
    document.getElementById('gradcamImage').src = data.images.gradcam;
    
    const severityBar = document.getElementById('severityBar');
    const severityLevel = data.severity.level;
    const severityLabel = data.severity.label;
    
    severityBar.style.width = '0%';
    severityBar.className = 'severity-bar';
    
    if (severityLevel === 0) severityBar.classList.add('healthy');
    else if (severityLevel < 50) severityBar.classList.add('mild');
    else if (severityLevel < 70) severityBar.classList.add('moderate');
    else if (severityLevel < 85) severityBar.classList.add('severe');
    else severityBar.classList.add('critical');
    
    setTimeout(() => {
        severityBar.style.width = severityLevel + '%';
        severityBar.textContent = severityLevel + '%';
    }, 100);
    
    document.getElementById('severityText').textContent = `${severityLevel}% - ${severityLabel}`;
    
    const treatmentList = document.getElementById('treatmentList');
    treatmentList.innerHTML = '';
    data.treatment.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        treatmentList.appendChild(li);
    });
    
    const preventionList = document.getElementById('preventionList');
    preventionList.innerHTML = '';
    data.prevention.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        preventionList.appendChild(li);
    });
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

window.addEventListener('load', async () => {
    try {
        console.log('Testing backend...');
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        console.log('✅ Backend status:', data);
    } catch (error) {
        console.error('⚠️ Backend not running:', error);
    }
});