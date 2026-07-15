import React from 'react';

const Suggestions = ({ onSelect, disabled = false }) => {
  const suggestions = [
    'How does solar energy work?',
    'Benefits of wind power',
    'Cost of renewable energy',
    'Environmental impact'
  ];

  return (
    <div className="suggestions">
      <p>Quick Questions:</p>
      <div className="suggestions-grid">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            type="button"
            onClick={() => onSelect(suggestion)}
            disabled={disabled}
            className="suggestion-btn"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default Suggestions;
