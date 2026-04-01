import React from 'react';


/* Static disclaimer entries rendered as individual pill-box cards. */
const DISCLAIMERS = [
  {
    id: 1,
    text: 'Do not rely on this application for medical diagnosis. These tools are used for research and testing.',
  },
  {
    id: 2,
    text: 'If you believe you may need medical attention, please seek professional advice and help.',
  },
  {
    id: 3,
    text: "This application is not a replacement for a doctor's opinion or diagnosis. Please consult a doctor if you are unsure or uncertain about a condition.",
  },
  {
    id: 4,
    text: 'Our models achieve an average accuracy of approximately 80%-85%. The goal of this application is to test the viability for skin diseases and skin cancer screening automation. The output and data from these models are not medical diagnoses and should not be treated as a replacement for a dermatologist. Always seek medical attention if you believe you may have a condition.',
  },
  {
    id: 5,
    text: 'This application uses deep learning models trained by the team and can make mistakes during inference.',
  },
];


/* The following function is responsible for rendering the warning triangle SVG
   icon displayed at the top of each disclaimer card. */
function WarningIcon() {

  return (
    <svg
      width="30"
      height="28"
      viewBox="0 0 30 28"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M15 2L28 25H2L15 2Z"
        stroke="#C8706E"
        strokeWidth="2.2"
        strokeLinejoin="round"
        fill="none"
      />
      <line
        x1="15"
        y1="11"
        x2="15"
        y2="18"
        stroke="#C8706E"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <circle cx="15" cy="22" r="1.4" fill="#C8706E" />
    </svg>
  );

}


/* The following function is responsible for rendering an individual disclaimer
   card containing the warning icon and the disclaimer text content. */
function DisclaimerCard({ text }) {

  return (
    <div className="disclaimer-card">
      <div className="disclaimer-icon-wrap">
        <WarningIcon />
      </div>
      <p className="disclaimer-text">{text}</p>
    </div>
  );

}


/* The following function is responsible for rendering the disclaimers page,
   displaying all five disclaimer entries in a side-by-side pill-box grid
   layout with glow shadow styling. */
function DisclaimersPage() {

  return (
    <div className="disclaimers-page">

      <div className="page-header">
        <h1 className="page-title">Disclaimers</h1>
        <p className="page-subtitle">
          Read and understand all notices before using any diagnostic tool in this application.
        </p>
      </div>

      <div className="disclaimers-grid">
        {DISCLAIMERS.map((item) => (
          <DisclaimerCard key={item.id} text={item.text} />
        ))}
      </div>

    </div>
  );

}

export default DisclaimersPage;
