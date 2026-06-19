import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { Explorer } from './Explorer';
import './styles.css';

// Lightweight routing: /explorer (or ?view=explorer) renders the log explorer
// as a standalone view (intended to open in its own window/tab).
const path = window.location.pathname.replace(/\/+$/, '');
const isExplorer = path.endsWith('/explorer')
  || new URLSearchParams(window.location.search).get('view') === 'explorer';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {isExplorer ? <Explorer /> : <App />}
  </React.StrictMode>,
);
