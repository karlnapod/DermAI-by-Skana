import React from 'react';
import { Link } from 'react-router-dom';


/* The following function is responsible for rendering the 404 Not Found page
   that is displayed when the user navigates to any route that does not match
   a defined application path. It provides a clear error message and a direct
   link back to the landing page. */
function NotFoundPage() {

  return (
    <div className="not-found-page">

      <p className="not-found-code">404</p>

      <h1 className="not-found-heading">Page Not Found</h1>

      <p className="not-found-body">
        The page you are looking for does not exist or may have been moved.
      </p>

      <Link to="/" className="not-found-btn">
        Back to Home
      </Link>

    </div>
  );

}

export default NotFoundPage;
