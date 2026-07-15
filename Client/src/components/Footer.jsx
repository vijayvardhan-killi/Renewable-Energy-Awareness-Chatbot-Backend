import React from 'react';
import { Zap } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer>
      <div className="footer-container">
        <div className="footer-content">
          <div>
            <div className="footer-brand">
              <div className="footer-brand-icon">
                <Zap />
              </div>
              <span>Green Genie</span>
            </div>
            <p>
              Helping everyone understand renewable energy through simple, practical answers.
            </p>
          </div>

          <div className="footer-section">
            <h3>Energy Topics</h3>
            <ul className="footer-links">
              <li><a href="#">Solar Energy</a></li>
              <li><a href="#">Wind Power</a></li>
              <li><a href="#">Hydroelectric</a></li>
              <li><a href="#">Geothermal</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h3>Platform</h3>
            <ul className="footer-links">
              <li><a href="#">Chat Assistant</a></li>
              <li><a href="#">Learning Resources</a></li>
              <li><a href="#">FAQ</a></li>
              <li><a href="#">Contact</a></li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom">
          <p>© {currentYear} Green Genie. All rights reserved.</p>
          <div className="footer-bottom-links">
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
