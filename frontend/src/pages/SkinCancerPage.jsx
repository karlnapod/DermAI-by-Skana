import React, { useState, useRef, useEffect } from 'react';


/* ------------------------------------------------------------------ */
/*  Static content                                                      */
/* ------------------------------------------------------------------ */

const RESULT_CONTENT = {
  benign: {
    title: 'Benign',
    body:  'The AI model classified the image as benign and cancer free.',
    disclaimer:
      'Disclaimer: Do not rely on the information provided by the AI model ' +
      'for professional medical advice. Always consult a doctor for skin ' +
      'cancer detection.',
  },
  malignant: {
    title: 'Malignant',
    body:
      'The AI model classified the image as malignant. Consult a ' +
      'dermatologist immediately as this is a severe diagnosis and will ' +
      'require serious medical attention.',
    disclaimer:
      'Disclaimer: Do not rely on the information provided by the AI model ' +
      'for professional medical advice. Always consult a doctor for skin ' +
      'cancer detection.',
  },
  inconclusive: {
    title: 'Inconclusive',
    body:
      'The AI model failed to classify the image as either benign or ' +
      'malignant. The confidence in the prediction was below the minimum ' +
      'threshold required to report a result.',
    disclaimer:
      'Disclaimer: Do not rely on the information provided by the AI model ' +
      'for professional medical advice. Always consult a doctor for skin ' +
      'cancer detection.',
  },
};

const MODEL_ARCHITECTURE_STATS = [
  { label: 'Architecture',     value: 'ResNet-50' },
  { label: 'Parameters',       value: '~25.6 M' },
  { label: 'Classifier head',  value: '2-layer MLP' },
  { label: 'Task type',        value: 'Binary' },
];

const MODEL_PERFORMANCE_STATS = [
  { label: 'Test accuracy',     value: '~86 %' },
  { label: 'Sensitivity',       value: '83.3 %' },
  { label: 'Specificity',       value: '88.6 %' },
  { label: 'Decision threshold', value: '0.50' },
];

const MODEL_IO_STATS = [
  { label: 'Input size',       value: '224 × 224 px' },
  { label: 'Channels',         value: 'RGB' },
  { label: 'Normalisation',    value: 'ImageNet' },
  { label: 'Accepted formats', value: 'JPG, PNG' },
];


/* ------------------------------------------------------------------ */
/*  Sub-components                                                      */
/* ------------------------------------------------------------------ */

/* The following function is responsible for rendering the SVG-based
   confidence ring that visually communicates the model's certainty.
   The ring animates from empty to the target fill on first render.
   If the prediction is inconclusive the ring is shown in dashed grey. */
function ConfidenceRing({ confidence, prediction }) {

  const RADIUS       = 52;
  const STROKE_WIDTH = 10;
  const PAD          = 6;
  const SIZE         = (RADIUS + STROKE_WIDTH / 2 + PAD) * 2;
  const CENTER       = SIZE / 2;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

  const [animatedOffset, setAnimatedOffset] = useState(CIRCUMFERENCE);

  /* The following effect is responsible for triggering the ring fill
     animation on the frame after the result is first rendered. */
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      if (prediction === 'inconclusive') {
        setAnimatedOffset(CIRCUMFERENCE);
      } else {
        setAnimatedOffset(CIRCUMFERENCE * (1 - confidence / 100));
      }
    });
    return () => cancelAnimationFrame(id);
  }, [confidence, prediction, CIRCUMFERENCE]);

  const isInconclusive = prediction === 'inconclusive';

  const ringColor =
    prediction === 'malignant' ? '#C8706E'
    : prediction === 'benign'  ? '#2AADA8'
    :                            '#8A9299';

  return (
    <div className="confidence-ring-wrap">

      <svg
        width={SIZE}
        height={SIZE}
        className="confidence-ring-svg"
        aria-hidden="true"
      >
        {/* Background track */}
        <circle
          cx={CENTER}
          cy={CENTER}
          r={RADIUS}
          fill="none"
          stroke="rgba(197, 205, 209, 0.38)"
          strokeWidth={STROKE_WIDTH}
          strokeDasharray={isInconclusive ? '8 7' : undefined}
        />

        {/* Animated fill arc — hidden when inconclusive */}
        {!isInconclusive && (
          <circle
            cx={CENTER}
            cy={CENTER}
            r={RADIUS}
            fill="none"
            stroke={ringColor}
            strokeWidth={STROKE_WIDTH}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={animatedOffset}
            strokeLinecap="round"
            transform={`rotate(-90 ${CENTER} ${CENTER})`}
            style={{
              transition: 'stroke-dashoffset 1.1s cubic-bezier(0.34, 1.56, 0.64, 1)',
            }}
          />
        )}
      </svg>

      <div className="confidence-ring-center">
        {isInconclusive ? (
          <span className="ring-inconclusive-label">Results{'\n'}Inconclusive</span>
        ) : (
          <>
            <span className={`ring-percentage ${prediction}`}>
              {confidence.toFixed(1)}%
            </span>
            <span className="ring-sub-label">Confidence</span>
          </>
        )}
      </div>

    </div>
  );

}


/* The following function is responsible for rendering the labelled result
   card that describes the classification outcome in plain language. */
function ResultCard({ prediction }) {

  const content = RESULT_CONTENT[prediction];

  return (
    <div className={`result-card ${prediction}`}>
      <p className={`result-card-title ${prediction}`}>{content.title}</p>
      <p className="result-card-body">{content.body}</p>
      <p className="result-card-disclaimer">{content.disclaimer}</p>
    </div>
  );

}


/* The following function is responsible for rendering the full results
   pill, which contains the confidence ring on the left and the result
   card on the right, separated by a vertical divider. */
function ResultsPill({ result }) {

  return (
    <div className="results-pill">

      <div className="results-pill-left">
        <ConfidenceRing
          confidence={result.confidence}
          prediction={result.prediction}
        />
        <span className="ring-caption">Model Confidence</span>
      </div>

      <div className="results-pill-divider" aria-hidden="true" />

      <div className="results-pill-right">
        <ResultCard prediction={result.prediction} />
      </div>

    </div>
  );

}


/* The following function is responsible for rendering one statistical
   row inside a model information card. */
function ModelStat({ label, value }) {

  return (
    <div className="model-info-stat">
      <span className="model-info-stat-label">{label}</span>
      <span className="model-info-stat-value">{value}</span>
    </div>
  );

}


/* The following function is responsible for rendering the model
   architecture, performance, and I/O specification cards at the
   bottom of the cancer detector page. */
function ModelInfoSection() {

  return (
    <div className="model-info-section">

      <h2 className="model-info-heading">Model Transparency</h2>
      <p className="model-info-description">
        The following specifications describe the deep learning model used
        for classification. All results produced by this model are for
        research purposes only and do not constitute a medical diagnosis.
      </p>

      <div className="model-info-grid">

        <div className="model-info-card">
          <p className="model-info-card-title">Architecture</p>
          {MODEL_ARCHITECTURE_STATS.map((s) => (
            <ModelStat key={s.label} label={s.label} value={s.value} />
          ))}
        </div>

        <div className="model-info-card">
          <p className="model-info-card-title">Performance</p>
          {MODEL_PERFORMANCE_STATS.map((s) => (
            <ModelStat key={s.label} label={s.label} value={s.value} />
          ))}
        </div>

        <div className="model-info-card">
          <p className="model-info-card-title">Input / Output</p>
          {MODEL_IO_STATS.map((s) => (
            <ModelStat key={s.label} label={s.label} value={s.value} />
          ))}
        </div>

      </div>

    </div>
  );

}


/* ------------------------------------------------------------------ */
/*  Main page component                                                 */
/* ------------------------------------------------------------------ */

/* The following function is responsible for rendering the Skin Cancer
   Detector page, managing the full upload → inference → results flow,
   and resetting state when the user requests a new scan. */
function SkinCancerPage() {

  const [selectedFile, setSelectedFile]   = useState(null);
  const [previewUrl,   setPreviewUrl]     = useState(null);
  const [isDragOver,   setIsDragOver]     = useState(false);
  const [isLoading,    setIsLoading]      = useState(false);
  const [result,       setResult]         = useState(null);
  const [error,        setError]          = useState(null);

  const fileInputRef = useRef(null);


  /* The following function is responsible for cleaning up the previous
     object URL and loading a new file into preview state. */
  function loadFile(file) {

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);

  }


  /* The following function is responsible for handling drag-over events
     on the upload zone to provide visual drop-target feedback. */
  function handleDragOver(e) {
    e.preventDefault();
    setIsDragOver(true);
  }


  /* The following function is responsible for resetting the drag-over
     state when the dragged item leaves the upload zone boundary. */
  function handleDragLeave() {
    setIsDragOver(false);
  }


  /* The following function is responsible for receiving a dropped file,
     validating the MIME type, and loading it into preview state. */
  function handleDrop(e) {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
      loadFile(file);
    }
  }


  /* The following function is responsible for reading the file chosen
     through the native file picker dialog and loading it into preview. */
  function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) loadFile(file);
  }


  /* The following function is responsible for programmatically opening
     the hidden file input to trigger the system file picker. */
  function triggerFileInput() {
    fileInputRef.current.click();
  }


  /* The following function is responsible for submitting the selected
     image to the FastAPI cancer detection endpoint and storing the
     inference result in component state. */
  async function runAnalysis() {

    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {

      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('/api/predict/skin-cancer', {
        method: 'POST',
        body:   formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Request failed.' }));
        throw new Error(err.detail || 'Analysis failed.');
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }

  }


  /* The following function is responsible for resetting all page state
     to the initial upload view so the user can scan a new image. */
  function resetPage() {

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setIsLoading(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

  }


  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div className="cancer-page">

      <div className="page-header">
        <h1 className="page-title">Skin Cancer Detector</h1>
        <p className="page-subtitle">
          Powered by a custom-trained deep learning classification model.
        </p>
      </div>

      {/* Hidden native file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png"
        className="upload-file-input"
        onChange={handleFileSelect}
      />

      {/* Upload zone — shown only when no file is selected */}
      {!selectedFile && (
        <div
          className={isDragOver ? 'upload-zone dragover' : 'upload-zone'}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <p className="upload-zone-title">Drag and drop an image here</p>
          <p className="upload-zone-sub">Supports JPG and PNG formats</p>
          <button
            className="upload-zone-btn"
            onClick={(e) => { e.stopPropagation(); triggerFileInput(); }}
          >
            Upload from Files
          </button>
        </div>
      )}

      {/* Image preview + action area — shown once a file is selected */}
      {selectedFile && (
        <div className="image-preview-section">

          <div className="image-preview-frame">
            <img
              src={previewUrl}
              alt="Uploaded dermoscopic image"
              className="image-preview-img"
            />
          </div>

          {/* Action row — only shown before inference runs */}
          {!result && !isLoading && (
            <>
              <button
                className="analyse-btn"
                onClick={runAnalysis}
                disabled={isLoading}
              >
                Analyse Image
              </button>
              <button className="change-image-btn" onClick={triggerFileInput}>
                Choose a different image
              </button>
            </>
          )}

          {/* Loading state */}
          {isLoading && (
            <div className="loading-indicator">
              <span className="spinner" />
              Analysing image, please wait...
            </div>
          )}

          {/* Error message */}
          {error && (
            <p className="error-msg">{error}</p>
          )}

          {/* Results section */}
          {result && (
            <div className="results-section">
              <ResultsPill result={result} />
              <button className="scan-new-btn" onClick={resetPage}>
                Scan A New Image
              </button>
            </div>
          )}

        </div>
      )}

      <ModelInfoSection />

    </div>
  );

}

export default SkinCancerPage;
