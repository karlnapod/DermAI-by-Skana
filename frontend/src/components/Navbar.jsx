import React from 'react';
import { NavLink } from 'react-router-dom';


/* The following function is responsible for rendering the persistent top-level
   navigation bar with the application logo acting as a home link and the four
   main section navigation buttons. */
function Navbar() {

  return (
    <nav className="navbar">

      <NavLink to="/" className="navbar-logo" aria-label="DermAI by Skana - Home">
        <img src="/assets/images/logo.png" alt="DermAI by Skana" />
        <span className="navbar-brand-name">DermAI by Skana</span>
      </NavLink>

      <div className="navbar-links">

        <NavLink
          to="/disclaimers"
          className={({ isActive }) => isActive ? 'navbar-link active' : 'navbar-link'}
        >
          Disclaimers
        </NavLink>

        <NavLink
          to="/skin-disease"
          className={({ isActive }) => isActive ? 'navbar-link active' : 'navbar-link'}
        >
          Skin Disease Detector
        </NavLink>

        <NavLink
          to="/skin-cancer"
          className={({ isActive }) => isActive ? 'navbar-link active' : 'navbar-link'}
        >
          Skin Cancer Detector
        </NavLink>

        <NavLink
          to="/about"
          className={({ isActive }) => isActive ? 'navbar-link active' : 'navbar-link'}
        >
          About Us
        </NavLink>

      </div>

    </nav>
  );

}

export default Navbar;
