import React from 'react';


/* The following function is responsible for rendering the main landing page,
   displaying the application hero title and the disclaimer notice subtitle
   centered in the viewport. */
function LandingPage() {

  return (
    <section className="landing-hero">

      <h1 className="hero-title">
        Your AI Skin Health Diagnostic Specialist
      </h1>

      <p className="hero-subtitle">
        Please ensure the disclaimers are read and understood before
        proceeding to use any tools in this application.
      </p>

    </section>
  );

}

export default LandingPage;
