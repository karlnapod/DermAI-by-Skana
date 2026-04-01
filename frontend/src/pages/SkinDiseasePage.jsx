import React, { useState, useRef, useEffect } from 'react';


/* ------------------------------------------------------------------ */
/*  Static metadata                                                     */
/* ------------------------------------------------------------------ */

/* Class metadata — abbreviations, full names, and CSS color class tokens.
   Order and abbreviations must match CLASS_ORDER in disease_classifier.py. */
const CLASS_META = {
  MEL:  { label: 'Melanoma',                color: 'mel'  },
  NV:   { label: 'Melanocytic Nevus',        color: 'nv'   },
  BCC:  { label: 'Basal Cell Carcinoma',     color: 'bcc'  },
  AK:   { label: 'Actinic Keratosis',        color: 'ak'   },
  BKL:  { label: 'Benign Keratosis',         color: 'bkl'  },
  DF:   { label: 'Dermatofibroma',           color: 'df'   },
  VASC: { label: 'Vascular Lesion',          color: 'vasc' },
  SCC:  { label: 'Squamous Cell Carcinoma',  color: 'scc'  },
};

/* Model transparency statistics sourced from disease_model/summary.json
   and disease_model/config.json. */
const MODEL_ARCHITECTURE_STATS = [
  { label: 'Architecture',     value: 'ResNet-152' },
  { label: 'Parameters',       value: '~60.2 M' },
  { label: 'Classifier head',  value: 'Single linear layer' },
  { label: 'Task type',        value: '8-class classification' },
];

const MODEL_PERFORMANCE_STATS = [
  { label: 'Test accuracy',    value: '86.2 %' },
  { label: 'Macro F1',         value: '81.1 %' },
  { label: 'Macro precision',  value: '86.5 %' },
  { label: 'Best epoch',       value: '27 / 100' },
];

const MODEL_IO_STATS = [
  { label: 'Input size',       value: '384 × 384 px' },
  { label: 'Channels',         value: 'RGB' },
  { label: 'Normalisation',    value: 'ImageNet' },
  { label: 'Accepted formats', value: 'JPG, PNG' },
];


/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */

/* The following function is responsible for mapping a numeric severity
   score to a human-readable tier label and a CSS modifier class that
   drives the severity bar fill colour. */
function getSeverityTier(score) {

  if (score >= 80) return { label: 'Critical — Immediate Attention Required', tierClass: 'critical' };
  if (score >= 60) return { label: 'High — Consult a Dermatologist Soon',     tierClass: 'high'     };
  if (score >= 40) return { label: 'Moderate — Seek Medical Advice',          tierClass: 'moderate' };
  if (score >= 20) return { label: 'Low-Moderate — Monitor and Review',       tierClass: 'lowmod'   };

  return                  { label: 'Low — Periodic Monitoring Sufficient',    tierClass: 'low'      };

}


/* ------------------------------------------------------------------ */
/*  Sub-components                                                      */
/* ------------------------------------------------------------------ */

/* The following function is responsible for rendering the animated
   horizontal probability bar for a single disease class.
   The bar width transitions from 0 to the target percentage on mount. */
function ClassBar({ abbrev, percentage, animate }) {

  const [width, setWidth] = useState(0);
  const meta = CLASS_META[abbrev] ?? { label: abbrev, color: 'nv' };

  useEffect(() => {

    if (!animate) {
      setWidth(0);
      return;
    }

    const id = requestAnimationFrame(() => setWidth(percentage));
    return () => cancelAnimationFrame(id);

  }, [percentage, animate]);

  return (
    <div className="class-bar-item">

      <div className="class-bar-label-row">
        <div className="class-bar-name-group">
          <span className="class-bar-abbrev">{abbrev}</span>
          <span className="class-bar-label">{meta.label}</span>
        </div>
        <span className="class-bar-pct">{percentage.toFixed(1)}%</span>
      </div>

      <div className="class-bar-track">
        <div
          className={`class-bar-fill ${meta.color}`}
          style={{ width: `${width}%` }}
        />
      </div>

    </div>
  );

}


/* The following function is responsible for rendering all 8 disease class
   probability bars sorted by descending probability score. */
function ClassBarsSection({ probabilities, animate }) {

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="class-bars-section">

      <p className="class-bars-heading">Class Probability Distribution</p>

      <div className="class-bars-list">
        {sorted.map(([abbrev, pct]) => (
          <ClassBar
            key={abbrev}
            abbrev={abbrev}
            percentage={pct}
            animate={animate}
          />
        ))}
      </div>

    </div>
  );

}


/* The following function is responsible for rendering the animated severity
   assessment bar that communicates the medical urgency level of the
   predicted class using a colour-coded fill and tier label. */
function SeverityBar({ severity, animate }) {

  const [width, setWidth] = useState(0);
  const tier = getSeverityTier(severity);

  useEffect(() => {

    if (!animate) {
      setWidth(0);
      return;
    }

    const id = requestAnimationFrame(() => setWidth(severity));
    return () => cancelAnimationFrame(id);

  }, [severity, animate]);

  return (
    <div className="severity-bar-section">

      <div className="severity-bar-label-row">
        <span className="severity-bar-heading">Severity Assessment</span>
        <span className="severity-bar-pct">{severity}%</span>
      </div>

      <div className="severity-bar-track">
        <div
          className={`severity-bar-fill ${tier.tierClass}`}
          style={{ width: `${width}%` }}
        />
      </div>

      <p className="severity-tier-label">{tier.label}</p>

      <p className="severity-disclaimer">
        Disclaimer: Do not rely on the information provided by the AI model
        for professional medical advice. Always consult a doctor if you
        believe there is a cause for concern.
      </p>

    </div>
  );

}


/* The following function is responsible for rendering the result pill
   container that displays the top predicted class name on the left and
   the severity assessment bar on the right, separated by a divider. */
function DiseaseResultPill({ result, animate }) {

  const meta = CLASS_META[result.predicted_class] ?? {};

  return (
    <div className="disease-result-pill">

      <div className="disease-result-left">
        <p className="disease-result-eyebrow">Predicted Condition</p>
        <p className={`disease-result-class-name ${meta.color ?? ''}`}>
          {result.predicted_description}
        </p>
        <p className="disease-result-confidence">
          Model confidence: {result.confidence.toFixed(1)}%
        </p>
      </div>

      <div className="results-pill-divider" aria-hidden="true" />

      <div className="disease-result-right">
        <SeverityBar severity={result.severity} animate={animate} />
      </div>

    </div>
  );

}


/* The following function is responsible for rendering the AI-generated
   description section for a disease classification result. It enforces a
   rate limit of 5 descriptions per 2-minute window (tracked via a ref
   passed from the parent so the limit persists across page resets).
   The "Generate Description" button must be clicked manually — the API is
   never called automatically on inference. */
function LLMDescriptionSection({ result, rateLimitRef, onDescriptionGenerated }) {

  const LIMIT     = 5;
  const WINDOW_MS = 2 * 60 * 1000;

  const [description,  setDescription]  = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [genError,     setGenError]     = useState(null);
  const [,             setTick]         = useState(0);

  const now               = Date.now();
  const recentTimestamps  = rateLimitRef.current.filter(t => now - t < WINDOW_MS);
  const remaining         = Math.max(0, LIMIT - recentTimestamps.length);
  const isRateLimited     = remaining === 0;
  const oldestTs          = isRateLimited ? Math.min(...recentTimestamps) : null;
  const secsUntilReset    = isRateLimited
    ? Math.max(0, Math.ceil((oldestTs + WINDOW_MS - now) / 1000))
    : 0;

  /* The following effect is responsible for triggering a re-render every
     second while rate-limited so the countdown display stays current. */
  useEffect(() => {
    if (!isRateLimited) return;
    const id = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(id);
  }, [isRateLimited]);


  /* The following function is responsible for calling the backend LLM
     descriptor endpoint, updating the rate limit tracker before the
     request, and storing the returned description in component state. */
  async function handleGenerate() {

    if (isRateLimited || isGenerating) return;

    const ts = Date.now();
    rateLimitRef.current = [
      ...rateLimitRef.current.filter(t => ts - t < WINDOW_MS),
      ts,
    ];

    setIsGenerating(true);
    setGenError(null);

    try {

      const response = await fetch('/api/llm/describe-disease', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          predicted_class:       result.predicted_class,
          predicted_description: result.predicted_description,
          confidence:            result.confidence,
          severity:              result.severity,
          probabilities:         result.probabilities,
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Request failed.' }));
        throw new Error(err.detail || 'Description generation failed.');
      }

      const data = await response.json();
      setDescription(data.description);
      // Notify parent so the PDF button can activate
      if (onDescriptionGenerated) onDescriptionGenerated(data.description);

    } catch (err) {
      setGenError(err.message);
    } finally {
      setIsGenerating(false);
    }

  }

  const countdownMins = Math.floor(secsUntilReset / 60);
  const countdownSecs = String(secsUntilReset % 60).padStart(2, '0');

  const rateClass = remaining === 0 ? ' depleted' : remaining <= 2 ? ' low' : '';

  return (
    <div className="llm-box">

      <div className="llm-box-header">
        <p className="llm-box-title">AI Description</p>
        <span className={`llm-rate-count${rateClass}`}>
          {remaining}/{LIMIT} remaining
        </span>
      </div>

      <div className="llm-box-body">

        {/* Default state — show generate button or rate limit message */}
        {!description && !isGenerating && (
          <div className="llm-generate-state">
            {isRateLimited ? (
              <p className="llm-rate-limit-msg">
                Generation limit reached. Resets in {countdownMins}m {countdownSecs}s.
              </p>
            ) : (
              <button className="llm-generate-btn" onClick={handleGenerate}>
                Generate Description
              </button>
            )}
            {genError && (
              <p className="llm-gen-error">{genError}</p>
            )}
          </div>
        )}

        {/* Loading state */}
        {isGenerating && (
          <div className="llm-generating-state">
            <span className="spinner" />
            <span className="llm-generating-text">Generating description...</span>
          </div>
        )}

        {/* Generated description */}
        {description && !isGenerating && (
          <p className="llm-description-text">{description}</p>
        )}

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


/* The following function is responsible for rendering the model architecture,
   performance, and I/O specification cards at the bottom of the disease
   detector page. Values are sourced from disease_model/summary.json and
   disease_model/config.json. */
function ModelInfoSection() {

  return (
    <div className="model-info-section">

      <h2 className="model-info-heading">Model Transparency</h2>
      <p className="model-info-description">
        The following specifications describe the deep learning model used
        for skin disease classification. All results produced by this model
        are for research purposes only and do not constitute a medical diagnosis.
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

/* The following function is responsible for rendering the Skin Disease
   Detector page, managing the full upload → inference → results flow,
   and resetting state when the user requests a new scan.

   RATE LIMITING (frontend side) — not yet implemented.
   In Phase 4, consider adding a client-side request guard to disable the
   Analyse button for a short cooldown period after each submission.
   This complements the server-side rate limiter defined in main.py. */
function SkinDiseasePage() {

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl,   setPreviewUrl]   = useState(null);
  const [isDragOver,   setIsDragOver]   = useState(false);
  const [isLoading,    setIsLoading]    = useState(false);
  const [result,       setResult]       = useState(null);
  const [error,        setError]        = useState(null);
  const [animateBars,  setAnimateBars]  = useState(false);

  // reportDescription is lifted here so the PDF button can read it
  // and resetPage can clear it when the user starts a new scan.
  const [reportDescription, setReportDescription] = useState(null);
  const [isPdfLoading,      setIsPdfLoading]      = useState(false);

  const fileInputRef       = useRef(null);
  // Rate limit ref persists across page resets so the 5/2-min limit
  // cannot be bypassed by repeatedly clicking "Scan A New Image".
  const descRateLimitRef   = useRef([]);


  /* The following effect is responsible for triggering bar and severity
     animations whenever a new result arrives, with a short delay to allow
     React to complete the DOM update before the transition begins. */
  useEffect(() => {

    if (!result) return;

    setAnimateBars(false);
    const timer = setTimeout(() => setAnimateBars(true), 80);
    return () => clearTimeout(timer);

  }, [result]);


  /* The following function is responsible for cleaning up the previous
     object URL and loading a new file into preview state. */
  function loadFile(file) {

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setAnimateBars(false);

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
     image to the FastAPI skin disease endpoint and storing the inference
     result in component state. */
  async function runAnalysis() {

    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setAnimateBars(false);

    try {

      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('/api/predict/skin-disease', {
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

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setIsLoading(false);
    setAnimateBars(false);
    setReportDescription(null);

    if (fileInputRef.current) fileInputRef.current.value = '';

  }


  /* The following helper is responsible for converting a blob URL (created by
     URL.createObjectURL) into a base64 data URL that jsPDF can embed directly
     into the generated PDF without writing to disk. */
  async function blobUrlToBase64(url) {
    const res  = await fetch(url);
    const blob = await res.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror   = reject;
      reader.readAsDataURL(blob);
    });
  }


  /* The following function is responsible for determining the severity tier
     label and clinical note for a given numeric severity score so the PDF
     report can display a plain-language risk assessment. */
  function pdfSeverityTier(score) {
    if (score >= 80) return { label: 'Emergency',  note: 'Immediate medical attention is required.' };
    if (score >= 60) return { label: 'High Risk',  note: 'Consult a dermatologist promptly.' };
    if (score >= 40) return { label: 'Moderate',   note: 'Medical evaluation is recommended.' };
    if (score >= 20) return { label: 'Low',         note: 'Routine monitoring is advised.' };
    return               { label: 'Minimal',      note: 'Generally benign; standard care applies.' };
  }


  /* The following function is responsible for building a single-page A4 PDF
     report from the current disease classification result, the uploaded image,
     and the AI-generated description. It uses jsPDF (loaded dynamically to
     avoid bundling overhead) and auto-downloads the file on completion. */
  async function generatePDF() {

    if (!result || !reportDescription || isPdfLoading) return;

    setIsPdfLoading(true);

    try {

      const { jsPDF } = await import('jspdf');

      const doc         = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
      const pageW       = doc.internal.pageSize.getWidth();   // 210 mm
      const marginX     = 18;
      const contentW    = pageW - marginX * 2;                // 174 mm
      const teal        = [42,  173, 168];
      const dark        = [58,  61,  64];
      const medium      = [75,  80,  87];
      const subtle      = [138, 146, 153];
      const border      = [172, 183, 189];

      let y = 18;

      /* ---- Header ---- */
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(18);
      doc.setTextColor(...teal);
      doc.text('DermAI by Skana', marginX, y);
      y += 9;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(10);
      doc.setTextColor(...medium);
      doc.text('Skin Disease Classification Report', marginX, y);
      y += 5.5;

      doc.setFontSize(8);
      doc.setTextColor(...subtle);
      const dateStr = new Date().toLocaleDateString('en-GB', {
        year: 'numeric', month: 'long', day: 'numeric',
      });
      doc.text(`Generated: ${dateStr}`, marginX, y);
      y += 7;

      doc.setDrawColor(...teal);
      doc.setLineWidth(0.5);
      doc.line(marginX, y, pageW - marginX, y);
      y += 8;

      /* ---- Image ---- */
      const imgBase64  = await blobUrlToBase64(previewUrl);
      const imgFormat  = selectedFile.type.includes('png') ? 'PNG' : 'JPEG';
      const imgSize    = 58;
      const imgX       = marginX + (contentW - imgSize) / 2;

      doc.setDrawColor(...border);
      doc.setLineWidth(0.3);
      doc.rect(imgX - 1, y - 1, imgSize + 2, imgSize + 2);
      doc.addImage(imgBase64, imgFormat, imgX, y, imgSize, imgSize, undefined, 'MEDIUM');
      y += imgSize + 3;

      doc.setFontSize(7.5);
      doc.setTextColor(...subtle);
      doc.text('Uploaded Skin Lesion Image', pageW / 2, y, { align: 'center' });
      y += 9;

      /* ---- Prediction section ---- */
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7.5);
      doc.setTextColor(...teal);
      doc.text('MODEL PREDICTION', marginX, y);
      y += 5.5;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);
      doc.setTextColor(...dark);
      doc.text(`Predicted Condition:  ${result.predicted_description} (${result.predicted_class})`, marginX, y);
      y += 5;
      doc.text(`Confidence Score:  ${result.confidence}%`, marginX, y);
      y += 8;

      /* ---- Divider ---- */
      doc.setDrawColor(...border);
      doc.setLineWidth(0.25);
      doc.line(marginX, y, pageW - marginX, y);
      y += 7;

      /* ---- Severity section ---- */
      const tier = pdfSeverityTier(result.severity);

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7.5);
      doc.setTextColor(...teal);
      doc.text('SEVERITY ASSESSMENT', marginX, y);
      y += 5.5;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);
      doc.setTextColor(...dark);
      doc.text(`Severity Score:  ${result.severity} / 100`, marginX, y);
      y += 5;
      doc.text(`Risk Level:  ${tier.label}`, marginX, y);
      y += 5;

      doc.setFont('helvetica', 'italic');
      doc.setFontSize(8.5);
      doc.setTextColor(...medium);
      doc.text(tier.note, marginX, y);
      y += 9;

      /* ---- Divider ---- */
      doc.setDrawColor(...border);
      doc.setLineWidth(0.25);
      doc.line(marginX, y, pageW - marginX, y);
      y += 7;

      /* ---- AI Description section ---- */
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7.5);
      doc.setTextColor(...teal);
      doc.text('AI GENERATED DESCRIPTION', marginX, y);
      y += 6;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);
      doc.setTextColor(...dark);

      const descLines = doc.splitTextToSize(reportDescription, contentW);
      // Cap at 15 lines to guarantee single-page fit
      const capped    = descLines.slice(0, 15);
      doc.text(capped, marginX, y);
      y += capped.length * 4.6 + 8;

      /* ---- Footer divider ---- */
      doc.setDrawColor(...border);
      doc.setLineWidth(0.25);
      doc.line(marginX, y, pageW - marginX, y);
      y += 5;

      /* ---- Disclaimer ---- */
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7);
      doc.setTextColor(...subtle);
      doc.text('DISCLAIMER', marginX, y);
      y += 4.5;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7.5);
      const disclaimerText =
        'Do not rely on the information provided by the AI model for professional ' +
        'medical advice. Always consult a qualified dermatologist or medical ' +
        'professional if you have concerns about a skin condition. DermAI is an ' +
        'educational and research tool and does not provide clinical diagnosis ' +
        'or replace professional medical judgement.';
      const discLines = doc.splitTextToSize(disclaimerText, contentW);
      doc.text(discLines, marginX, y);

      /* ---- Save ---- */
      const safeName = result.predicted_class.toLowerCase();
      doc.save(`DermAI_Report_${safeName}.pdf`);

    } catch (err) {
      console.error('PDF generation failed:', err);
    } finally {
      setIsPdfLoading(false);
    }

  }


  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div className="disease-page">

      <div className="page-header">
        <h1 className="page-title">Skin Disease Detector</h1>
        <p className="page-subtitle">
          Powered by a ResNet-152 model trained on the ISIC 2019 dataset.
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
              alt="Uploaded skin lesion image"
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
            <div className="disease-results-section">

              <ClassBarsSection
                probabilities={result.probabilities}
                animate={animateBars}
              />

              <DiseaseResultPill result={result} animate={animateBars} />

              <LLMDescriptionSection
                result={result}
                rateLimitRef={descRateLimitRef}
                onDescriptionGenerated={setReportDescription}
              />

              <div className="disease-action-row">
                <button
                  className="pdf-btn"
                  onClick={generatePDF}
                  disabled={!reportDescription || isPdfLoading}
                  title={!reportDescription ? 'Generate an AI description first to unlock this report' : ''}
                >
                  {isPdfLoading ? 'Generating PDF...' : 'Generate PDF Report'}
                </button>

                <button className="scan-new-btn" onClick={resetPage}>
                  Scan A New Image
                </button>
              </div>

            </div>
          )}

        </div>
      )}

      <ModelInfoSection />

    </div>
  );

}

export default SkinDiseasePage;
