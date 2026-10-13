/* Frontend configuration.
 *
 * API_BASE = where the backend (FastAPI) runs.
 *   ""                                -> same server as this page (local run: python -m uvicorn app.main:app)
 *   "https://your-app.onrender.com"   -> hosted backend, when this page is on GitHub Pages
 *
 * You can also override it without editing this file by opening the page with
 *   ?api=https://your-app.onrender.com   (remembered in the browser)
 */
window.APP_CONFIG = {
  API_BASE: "",
};
