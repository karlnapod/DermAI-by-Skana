import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Navbar from './components/Navbar.jsx';
import ChatBot from './components/ChatBot.jsx';

import LandingPage from './pages/LandingPage.jsx';
import DisclaimersPage from './pages/DisclaimersPage.jsx';
import SkinDiseasePage from './pages/SkinDiseasePage.jsx';
import SkinCancerPage from './pages/SkinCancerPage.jsx';
import AboutPage from './pages/AboutPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';


/* The following function is responsible for rendering the ambient teal blob
   decorations that are fixed behind all page content to produce the soft
   blended background effect. */
function BackgroundBlobs() {

  return (
    <div className="bg-blobs" aria-hidden="true">
      <div className="bg-blob bg-blob-1" />
      <div className="bg-blob bg-blob-2" />
      <div className="bg-blob bg-blob-3" />
      <div className="bg-blob bg-blob-4" />
      <div className="bg-blob bg-blob-5" />
      <div className="bg-overlay" />
    </div>
  );

}


/* The following function is responsible for rendering the root application
   layout, combining client-side routing, the persistent navigation bar,
   background decoration, page content, and the global chatbot overlay. */
function App() {

  return (
    <Router>
      <div className="app-root">

        <BackgroundBlobs />

        <Navbar />

        <main className="app-content">
          <Routes>
            <Route path="/"              element={<LandingPage />}    />
            <Route path="/disclaimers"   element={<DisclaimersPage />} />
            <Route path="/skin-disease"  element={<SkinDiseasePage />} />
            <Route path="/skin-cancer"   element={<SkinCancerPage />}  />
            <Route path="/about"         element={<AboutPage />}        />
            <Route path="*"              element={<NotFoundPage />}     />
          </Routes>
        </main>

        <ChatBot />

      </div>
    </Router>
  );

}

export default App;
