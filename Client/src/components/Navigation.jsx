import React, { useState } from 'react';
import { Home, Bot, Info, Zap, Menu, X, Sun, Moon } from 'lucide-react';

const Navigation = ({ currentPage, navigate, darkMode, toggleDarkMode }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'chatbot', label: 'Chatbot', icon: Bot },
    { id: 'about', label: 'About', icon: Info }
  ];

  return (
    <nav>
      <div className="nav-container">
        <div className="nav-content">
          {/* Logo */}
          <div className="nav-logo">
            <div className="nav-logo-icon">
              <Zap />
            </div>
            <span>Green Genie</span>
          </div>

          {/* Desktop Navigation */}
          <div className="nav-desktop">
            {navItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => navigate(id)}
                className={`nav-button ${currentPage === id ? 'active' : ''}`}
              >
                <Icon size={16} />
                <span>{label}</span>
              </button>
            ))}
            <button onClick={toggleDarkMode} className="nav-button">
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>

          {/* Mobile menu button */}
          <div className="nav-mobile">
            <button onClick={toggleDarkMode} className="nav-button">
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="nav-button"
            >
              {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="nav-mobile-menu">
            <div className="nav-mobile-items">
              {navItems.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => {
                    navigate(id);
                    setIsMobileMenuOpen(false);
                  }}
                  className={`nav-button ${currentPage === id ? 'active' : ''}`}
                >
                  <Icon size={16} />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;
