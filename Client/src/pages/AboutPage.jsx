import React from 'react';

const AboutPage = () => {
  const technologies = [
    { name: 'React.js', description: 'Frontend framework for building interactive user interfaces' },
    { name: 'Custom CSS', description: 'Handcrafted styles for responsive and modern design' },
    { name: 'Lucide React', description: 'Icon library for beautiful and consistent iconography' },
    { name: 'AI Integration', description: 'Ready for ChatGPT, Dialogflow, or custom AI APIs' }
  ];

  const features = [
    'Interactive chatbot for personalized renewable energy advice',
    'Comprehensive information about solar, wind, hydro, and geothermal energy',
    'Responsive design that works on all devices',
    'Dark mode support for comfortable viewing',
    'Ready for integration with advanced AI services',
    'Educational content to raise awareness about clean energy'
  ];

  return (
    <div className="about-container">
      <div className="about-header">
        <h1>About Green Genie</h1>
        <p>
          Empowering communities with knowledge about renewable energy through intelligent conversations
        </p>
      </div>

      <div className="about-section">
        <h2>Our Mission</h2>
        <p>
          Climate change is one of the most pressing challenges of our time. The transition to renewable energy 
          is not just an environmental necessity, but an economic opportunity that can create jobs, reduce costs, 
          and improve quality of life for communities worldwide.
        </p>
        <p>
          Our platform aims to make renewable energy information accessible to everyone through an intelligent 
          chatbot that can answer questions, provide personalized recommendations, and help users understand 
          the benefits of clean energy solutions.
        </p>
      </div>

      <div className="about-section">
        <h2>How It's Built</h2>
        <div className="tech-grid">
          {technologies.map((tech, index) => (
            <div key={index} className="tech-card">
              <h3>{tech.name}</h3>
              <p>{tech.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="about-section">
        <h2>Key Features</h2>
        <div className="features-list">
          {features.map((feature, index) => (
            <div key={index} className="feature-item">
              <div className="feature-dot"></div>
              <p>{feature}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AboutPage;
