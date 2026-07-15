import React from 'react';

const RenewableCard = ({ icon: IconComponent, title, description, benefits }) => {
  return (
    <div className="renewable-card">
      <div className="renewable-icon">
        <IconComponent size={32} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <div className="benefits-list">
        {benefits.map((benefit, idx) => (
          <div key={idx} className="benefit-item">
            <div className="benefit-dot"></div>
            <span>{benefit}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RenewableCard; 
