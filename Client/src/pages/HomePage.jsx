import React from 'react';
import { Sun, Wind, Droplets, Zap, MessageCircle, Bot } from 'lucide-react';
import RenewableCard from "../components/RenewableCard";


const HomePage = ({ navigate }) => {
  const renewableTypes = [
    {
      title: 'Solar Energy',
      icon: Sun,
      description: 'Harness the power of the sun with photovoltaic panels and solar thermal systems.',
      benefits: ['Clean & Renewable', 'Reduces Electricity Bills', 'Low Maintenance']
    },
    {
      title: 'Wind Energy',
      icon: Wind,
      description: 'Generate electricity using wind turbines that convert kinetic energy from wind.',
      benefits: ['Sustainable', 'Cost-Effective', 'Creates Jobs']
    },
    {
      title: 'Hydroelectric',
      icon: Droplets,
      description: 'Use flowing water to generate clean electricity through hydroelectric dams.',
      benefits: ['Reliable Power', 'Long Lifespan', 'Flood Control']
    },
    {
      title: 'Geothermal',
      icon: Zap,
      description: 'Tap into Earth\'s natural heat for heating, cooling, and electricity generation.',
      benefits: ['24/7 Availability', 'Small Footprint', 'Very Reliable']
    }
  ];

  return (
    <div>
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-container">
          <div className="hero-content">
            <h1 className="hero-title">
              Power the Future with
              <span className="hero-title-highlight"> Renewable Energy</span>
            </h1>
            <p className="hero-description">
              Discover how clean, sustainable energy sources can transform our world and create a better tomorrow for generations to come.
            </p>
            <div className="hero-buttons">
              <button onClick={() => navigate('chatbot')} className="btn-primary">
                <MessageCircle size={20} />
                <span>Ask Green Genie, Your Energy Assistant</span>
              </button>
              <button onClick={() => navigate('about')} className="btn-secondary">Learn More</button>
            </div>
          </div>
        </div>
      </div>

      {/* Renewable Energy Types */}
      <div className="renewable-section">
        <div className="section-container">
          <div className="section-header">
            <h2 className="section-title">Types of Renewable Energy</h2>
            <p className="section-description">
              Explore the various clean energy sources that are revolutionizing how we power our world
            </p>
          </div>

          <div className="renewable-grid">
            {renewableTypes.map((type, index) => (
              <RenewableCard
                key={index}
                title={type.title}
                icon={type.icon}
                description={type.description}
                benefits={type.benefits}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="cta-section">
        <div className="section-container">
          <div className="cta-content">
            <h2>Ready to Learn More?</h2>
            <p>
              Chat with our AI assistant to get personalized information about renewable energy solutions
            </p>
            <button onClick={() => navigate('chatbot')} className="btn-primary">
              <Bot size={20} />
              <span>Start Chatting Now</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
